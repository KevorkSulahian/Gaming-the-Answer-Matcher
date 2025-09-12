import sys
import os
from pprint import pprint
from typing import Any, Literal
from datasets import load_dataset, Dataset, DatasetDict

import pandas as pd
import torch, gc

from inspect_ai.dataset import FieldSpec,csv_dataset
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

from inspect_ai import Task, eval, task

from inspect_ai.scorer import scorer, model_graded_fact, answer, choice, Scorer, Score, Target, accuracy, stderr
from inspect_ai.scorer import CORRECT, INCORRECT, Score
from inspect_ai.scorer._metrics import mean

from inspect_ai.solver import Solver, TaskState, chain, solver

from inspect_ai.log import read_eval_log
from inspect_ai import eval_set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpqa_diamond_baseline_answers_generation import generate_answers, answer_question_gpt
from answer_generation_prompts import UNSURE_MULTIPLE_ANSWERS, BASELINE_PROMPT
import answer_matching_prompts
from inference import HFInference

def init(testee_name):
    instance = HFInference(f"Qwen/{testee_name}")
    return instance

def exit(instance):
    del instance.model 
    del instance.tokenizer 
    del instance 
    gc.collect() 
    torch.cuda.empty_cache() 
      
def upload_to_hub(exprs, bench, testees):
    """
    Given generations for the experiments and bench for both qual and quant are available.

    testees: ["gpt", "qwen"]
    exprs = ["strategic", "baseline", "forward", "backward", "wrong"]
    """
    # 
    qual_df = pd.read_csv(f"../datasets/{bench}/{bench}_diamond_qualitative.csv")
    quant_df = pd.read_csv(f"../datasets/{bench}/{bench}_diamond_quantitative.csv")

    def get_splits(expr):
        if expr == "backward" or expr == "forward":
            base_path = "multiple" 
        else:
            base_path = expr

        splits = {}
        for testee_name in testees:
            qual_responses_qwen = pd.read_csv(f"../answer-generation/{bench}/{base_path}/{testee_name}_qual_{expr}_answers.csv")
            quant_responses_qwen = pd.read_csv(f"../answer-generation/{bench}/{base_path}/{testee_name}_quant_{expr}_answers.csv")
            qual_responses_qwen = get_resp_df(qual_df, qual_responses_qwen)
            quant_responses_qwen = get_resp_df(quant_df, quant_responses_qwen)

            qual_dataset = Dataset.from_pandas(qual_responses_qwen)
            quant_dataset = Dataset.from_pandas(quant_responses_qwen)
            splits[f"{testee_name}_qual"] = qual_dataset
            splits[f"{testee_name}_quant"] = quant_dataset

        strat = DatasetDict(splits)
        return strat

    experiments = {}
    for expr in exprs:
        splits = get_splits(expr)
        experiments[expr] = splits
        
    for name, ds in experiments.items():
        ds.push_to_hub("Sumana05/gaming_matchers", config_name=name, private=True)



def log_to_df(eval_log):
    try:
        df = pd.DataFrame([
            {
                'id': sample.id,
                'question': sample.input,
                'answer': sample.scores['match_scorer'].answer,
                'reference': sample.target,
                'score': sample.metadata['score'],
                'testee': eval_log.eval.model,
                'choices': sample.choices
            }
            for sample in eval_log.samples
        ])
    except:
        df = pd.DataFrame([
            {
                'id': sample.id,
                'question': sample.input,
                'answer': sample.output.completion,
                'reference': sample.target,
                'testee': eval_log.eval.model,
                'choices': sample.choices
            }
            for sample in eval_log.samples
        ])
    return df

def get_resp_df(q_df, a_df): 
    #original question df, generated responses df
    resp = q_df[['question', 'reference', 'question_mcq']].merge(a_df[['question', 'answer']], on='question')
    resp = resp[['question', 'reference', 'answer']]
    return resp

def get_gpqa_base_data():
    gpqa_qual_dataset = csv_dataset(
        "../datasets/gpqa/gpqa_diamond_qualitative.csv",
        FieldSpec(
            input="question",
            target="reference",
            choices="choices",
            metadata=["question_mcq"],
        ),
    )
    gpqa_quant_dataset = csv_dataset(
        "../datasets/gpqa/gpqa_diamond_quantitative.csv",
        FieldSpec(
            input="question",
            target="reference",
            choices="choices",
            metadata=["question_mcq"],
        ),
    )
    pprint(gpqa_qual_dataset.samples[1].__dict__)

    return gpqa_qual_dataset, gpqa_quant_dataset

def gpqa_record_to_sample(record: dict[str, Any]) -> Sample:

    target = record["reference"]
    input = [ChatMessageUser(content=record["question"])]  # should store input as list of ChatMessage objects
    answer = record["answer"]
    # return sample
    # 
    return Sample(input=input, target=target, metadata={"answer":answer})


def testee(testee_name, prompt, cache, cache_file, instance=None):
    @solver
    def _testee() -> Solver:
        async def solve(state: TaskState, generate) -> TaskState:
            #record
            ch = [choice.value for choice in state.choices]
            question = state.input
            # mcq = state.metadata['question_mcq']
            target = state.target.text
            record = {'question': question}
            id = state.sample_id
            # print()

            if "gpt" in testee_name.lower():
                text_response = answer_question_gpt(question, id, cache, cache_file, model=testee_name, PROMPT_TEMPLATE=prompt)
            if "qwen" in testee_name.lower():
                text_response = instance.generate_batch(record, cache, cache_file, prompt, system_prompt=None,
                       temperature=0.6, max_new_tokens=300)
            
            state.output.completion = text_response

            return state

        return solve     
      
    return _testee()




def matcher(matcher_name, prompt, cache, cache_file, instance=None, temperature=0.01, max_new_tokens=2048, num_tries=3):
    @solver
    def _matcher() -> Solver:
        async def solve(state: TaskState, generate) -> TaskState:
            #record
            ch = [choice.value for choice in state.choices]
            question = state.input
            target = state.target.text
            text_response = state.output.completion
            
            record = {'question': question, 'reference': target, 'answer': text_response, 
                      'choices': ", ".join(ch)}
            

            cache_key = f"{matcher_name}::{record['question']}"
            state.metadata["thinking"] = cache[cache_key]["reasoning"]

            if "gpt" in matcher_name.lower():
                text_response = answer_question_gpt(question, id, cache, cache_file, model=matcher_name, PROMPT_TEMPLATE=prompt)

            if "qwen" in matcher_name.lower():
                score = instance.generate_batch(record, cache, cache_file, prompt, system_prompt=None,
                       temperature=temperature, max_new_tokens=max_new_tokens)
                if len(score) > 1:
                    score = instance.regenerate_resp( record, num_tries, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                if len(score) > 1:
                    score = instance.regenerate_resp( record, num_tries, prompt, max_new_tokens=5000, temperature=temperature)
                else:
                    state.metadata['score'] = score
            


            return state

        return solve     
      
    return _matcher()



def only_matcher(matcher_name, prompt, cache, cache_file, instance=None, temperature=0.01, max_new_tokens=2048,
            num_tries=3):
    @solver
    def _only_matcher() -> Solver:
        async def solve(state: TaskState, generate) -> TaskState:
            #record
            ch = [choice.value for choice in state.choices]
            question = state.input
            target = state.target.text
            text_response = state.metadata["answer"]
            
            record = {'question': question, 'reference': target, 'answer': text_response, 
                      'choices': ", ".join(ch)}
            

            cache_key = f"{matcher_name}::{record['question']}"
            state.metadata["thinking"] = cache[cache_key]["reasoning"]
        

            if "gpt" in matcher_name.lower():
                text_response = answer_question_gpt(question, id, cache, cache_file, model=matcher_name, PROMPT_TEMPLATE=prompt)

            if "qwen" in matcher_name.lower():
                score = instance.generate_batch(record, cache, cache_file, prompt, system_prompt=None,
                       temperature=temperature, max_new_tokens=max_new_tokens)
                if len(score) > 1:
                    score = instance.regenerate_resp( record, num_tries, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                if len(score) > 1:
                    score = instance.regenerate_resp( record, num_tries, prompt, max_new_tokens=5000, temperature=temperature)
                else:
                    state.metadata['score'] = score


            return state

        return solve     
      
    return _only_matcher()


@scorer(metrics=[accuracy(), stderr()])
def match_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        try:
            matcher_op = int(state.metadata['score'].strip())
        except:
            matcher_op = state.metadata['score'].strip()
        # print(matcher_op)
        ans = state.output.completion
        
        return Score(
        value=CORRECT if matcher_op == 1 else INCORRECT,
        answer=ans,
        explanation=state.metadata['thinking'],
    )
    
    return score


def full_task_pipe():
    gpqa_qual_dataset, gpqa_quant_dataset = get_gpqa_base_data()

    cache_t = {}
    cache_m = {}
    expr = "baseline"
    bench = "gpqa"
    # testee_name = "gpt-4.1-mini"
    
    testee_name = 'qwen2.5-7b'
    cache_file_t = f"{bench}/{expr}/inspect_gen_{testee_name}.json"
    cache_file_m = f"{bench}/{expr}/inspect_match_{testee_name}.json"
    matcher_name = "qwen3-4b"
    inst_t = init(testee_name)
    inst_m = init(matcher_name)
    max_new_tokens = 2048
    temperature = 0.01
    num_tries = 3
    matcher_prompt = answer_matching_prompts.get_judge_prompt_with_gt_baseline()

    task = Task(
        dataset=gpqa_qual_dataset,
        plan=[
 
            testee(testee_name, UNSURE_MULTIPLE_ANSWERS, cache_t, cache_file_t, inst_t),
            exit(inst_t),
            matcher(matcher_name, matcher_prompt, cache_m, cache_file_m, instance=inst_m, temperature=temperature, max_new_tokens=max_new_tokens, num_tries=num_tries),
            exit(inst_m)
            ],
        scorer=[match_scorer()],
    )

    logm1 = eval(task, 
            model=f"openai/{testee_name}", log_dir = f"logs/full_eval/{testee_name}")
    eval_log = read_eval_log(logm1[0].location)
    print(eval_log.eval.task)
    print(eval_log.eval.dataset)
    print(eval_log.eval.model)
    df = log_to_df(eval_log)
    
    df.to_csv(f"{matcher_name}/{bench}_{expr}_{testee_name}_results.csv")


def generation_task(prompt, expr = "baseline", bench = "gpqa", btype = "qual", 
                    testee_name = 'qwen2.5-7b'):
    gpqa_qual_dataset, gpqa_quant_dataset = get_gpqa_base_data()
    cache_t = {}
    
    # testee_name = "gpt-4.1-mini"
    
    
    cache_file_t = f"{bench}/{expr}/inspect_gen_{testee_name}.json"
    inst_t = init(testee_name)

    task = Task(
        dataset=gpqa_qual_dataset,
        plan=[ 
            testee(testee_name, prompt, cache_t, cache_file_t, inst_t),
            exit(inst_t),
            ],
    )

    logm1 = eval(task, 
            model=f"openai/{testee_name}", log_dir = f"logs/generations/{testee_name}")
    eval_log = read_eval_log(logm1[0].location)
    print(eval_log.eval.task)
    print(eval_log.eval.dataset)
    print(eval_log.eval.model)
    df = log_to_df(eval_log)
    df.to_csv(f"../answer-generation/{bench}/{expr}/{testee_name}_{btype}_{expr}_answers.csv")

def matching_task(matcher_prompt, expr = "baseline", bench = "gpqa", btype = "qual", 
                  testee_name = 'qwen2.5-7b', matcher_name='qwen3-4b',
                  max_new_tokens = 2048, temperature = 0.01, num_tries = 3):

    # matcher_prompt = answer_matching_prompts.get_judge_prompt_with_gt_baseline()

    if "qwen" in testee_name:
        testee = "qwen"
    if "gpt" in testee_name:
        testee = "gpt"

    gpqa_ans_dset = hf_dataset(
    path=f"Sumana05/gaming_matchers",
    name=expr,
    split=f"{testee}_{btype}",                  
    sample_fields=gpqa_record_to_sample,
    )
    pprint(gpqa_ans_dset.samples[0].__dict__)

    cache = {}
    # testee_name = "gpt-4.1-mini"
    cache_file = f"{bench}/{expr}/inspect_match_{testee_name}.json"
    inst_m = init(matcher_name)
    

    task = Task(
        dataset=gpqa_ans_dset,
        plan=[ 
            only_matcher(matcher_name, matcher_prompt, cache, cache_file, instance=inst_m, temperature=temperature, max_new_tokens=max_new_tokens, num_tries=num_tries),
            exit(inst_m),
            ],
        scorer=[match_scorer()],
    )

    logm1 = eval(task, 
            model=f"openai/{testee_name}", log_dir = f"logs/matches/{testee_name}")
    eval_log = read_eval_log(logm1[0].location)
    print(eval_log.eval.task)
    print(eval_log.eval.dataset)
    print(eval_log.eval.model)
    df = log_to_df(eval_log)
    df.to_csv(f"{matcher_name}/{bench}_{expr}_{testee_name}_{btype}_matches.csv")

def main():
    bench = "gpqa"
    btypes = ["qual", "quant"]
    exprs = ['baseline']
    prompts = [BASELINE_PROMPT, ]
    testees = ["gpt", "qwen"]
    testee_names = ['gpt-4.1-mini', 'qwen2.5-7b']

    max_new_tokens = 2048
    temperature = 0.01
    num_tries = 3
    
    #could use eval_set
    for expr, prompt in dict(zip(exprs, prompts)):
        for btype in btypes:
            for testee_name in testee_names:
                generation_task(prompt, expr = expr, 
                                btype = btype, testee_name = testee_name)
            
    
    upload_to_hub(exprs, bench, testees)

    matcher_prompt = answer_matching_prompts.get_judge_prompt_with_gt_baseline()

    for expr in exprs:
        for btype in btypes:
            for testee_name in testee_names:
                matching_task(matcher_prompt, 
                              expr = expr, 
                              bench = bench, 
                              btype = btype, 
                              testee_name = testee_name, 
                              matcher_name ='qwen3-4b',
                              max_new_tokens = max_new_tokens, 
                              temperature = temperature, 
                              num_tries = num_tries)
        
if __name__ == "__main__":
    main() 