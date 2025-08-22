import os
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
import torch, gc

import prompts
from inference import HFInference

cache_lock = threading.Lock()

MAX_WORKERS = 8

def process_responses(
    responses_df,
    df_type,
    model,
    user_prompt,
    system_prompt=None,
    temperature=0.6,
    max_new_tokens=100,
    batch_size=4
):
    """
    Prompt for model judgement given question, answers and/or references

    responses_df:  records of question, reference, response
     df_type: qualitative/quantitative/val/test or whatever else
     model: model name not instance eg:Qwen2.5-7B
     user_prompt: user prompt to be formatted
     system_prompt: (optional)
     temperature: generation temp (optional) default:0.6, 
     max_new_tokens: num output tokens (optional) default:100

    """
    cache_file = f"gpqa_diamond_cache_{df_type}_{model}_matches.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    if "qwen" in model.lower():
        mod_inference = HFInference(f"Qwen/{model}")
    else:
        mod_inference = HFInference(model)

    results = []
    b_count = 0

    for start in range(0, len(responses_df), batch_size):
        batch = responses_df[start:start + batch_size]
        cache = mod_inference.generate_batch(
            batch, cache, cache_file, user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        for record in batch:
            cache_key = f"{mod_inference.model_name}::{record['question']}"
            results.append({
                'judge': model,
                'question': record['question'],
                'reference': record['reference'],
                'response': record['answer'],
                'score': cache[cache_key]['score']
            })
            print(cache[cache_key]['score'])
        b_count += 1
        print(f"Proc {b_count} batch")

    answer_df = pd.DataFrame(results, columns=["judge", "question", "reference", "response", "score"])
    
    del mod_inference.model 
    del mod_inference.tokenizer 
    del mod_inference 
    gc.collect() 
    torch.cuda.empty_cache() 
    print("Processing complete.")
    return answer_df

def get_resp_df(q_df, a_df):
    resp = q_df[['question', 'reference', 'question_mcq']].merge(a_df, on='question')
    resp = resp[['question', 'reference', 'answer']]
    return resp

def main():
    user_prompt_template = prompts.get_judge_prompt_with_gt_baseline() #default: with COT

    #baseline datasets, baseline responses, baseline prompt judgement
    qual_df = pd.read_csv("../datasets/gpqa/gpqa_diamond_qualitative.csv")
    quant_df = pd.read_csv("../datasets/gpqa/gpqa_diamond_quantitative.csv")

    qwen_responses = pd.read_csv("../answer-generation/gpqa/gpqa_diamond_qual_qwen_answers.csv")
    gpt_responses = pd.read_csv("../answer-generation/gpqa/gpqa_diamond_qual_gpt_answers.csv")
    qual_responses_qwen = get_resp_df(qual_df, qwen_responses)
    qual_responses_gpt = get_resp_df(qual_df, gpt_responses)

    qwen_responses = pd.read_csv("../answer-generation/gpqa/gpqa_diamond_quant_qwen_answers.csv")
    gpt_responses = pd.read_csv("../answer-generation/gpqa/gpqa_diamond_quant_gpt_answers.csv")
    quant_responses_qwen = get_resp_df(quant_df, qwen_responses)
    quant_responses_gpt = get_resp_df(quant_df, gpt_responses)


    qwen_qual_am_df = process_responses(qual_responses_qwen.to_dict(orient="records"), "qual", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    qwen_quant_am_df = process_responses(quant_responses_qwen.to_dict(orient="records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    qwen_qual_am_df.to_csv(f"gpqa_scores/gpqa_diamond_qual_qwen_matches_baseline.csv", index=False)
    qwen_quant_am_df.to_csv(f"gpqa_scores/gpqa_diamond_quant_qwen_matches_baseline.csv", index=False)

    gpt_qual_am_df = process_responses(qual_responses_gpt.to_dict(orient="records"), "qual", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    gpt_quant_am_df = process_responses(quant_responses_gpt.to_dict(orient="records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    gpt_qual_am_df.to_csv(f"gpqa_scores/gpqa_diamond_qual_gpt_matches_baseline.csv", index=False)
    gpt_quant_am_df.to_csv(f"gpqa_scores/gpqa_diamond_quant_gpt_matches_baseline.csv", index=False)


if __name__ == "__main__":
    main()