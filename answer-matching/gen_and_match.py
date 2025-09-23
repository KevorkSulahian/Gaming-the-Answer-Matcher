import os
import json
import re
import math
import pandas as pd
import random
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
import torch, gc
import warnings
warnings.filterwarnings('ignore')

import answer_matching_prompts


cache_lock = threading.Lock()

MAX_WORKERS = 8

def random_samples(input_list, max_sample_size):
    """Given a list of records/dicts, return a randomly sampled subset as list"""
    sample_size = min(max_sample_size, len(input_list))  # ensure not sampling more than available
    return random.sample(input_list, sample_size)

class HFInference:
    def __init__(self, model_name, device_map='auto', max_memory=None):
        print(f"Loading model {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device_map} if isinstance(device_map, str) and device_map.startswith("cuda") else device_map,
            max_memory=max_memory,
            trust_remote_code=True
        )
        print(f"Model {model_name} loaded on device: {self.model.device}")

        print("Model loaded successfully!")

    def extract_answer_tags(self, response):
        pattern = r'answer>\s*(?:>|>/)?\s*(.*?)\s*(?:</answer>|>/|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            num_match = re.search(r'\b\d+(\.\d+)?\b', content)
            if num_match:
                return num_match.group(0)
            return content
        return response


    def answer_question(self, record, question_number, cache, cache_file, prompt_template):
        """Generate answer using the pre-loaded model"""
        
        # Check cache first
        cache_key = f"qwen::{record['question']}"
        if cache_key in cache:
            print(f"[{question_number} cached]")
            return cache[cache_key]
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": prompt_template.format(**record)
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize and move to device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():  # Save memory
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=300,
                do_sample=True,  # Add some randomness
                temperature=0.6,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # print(content)
        # res = self.extract_answer_tags(content)
        res = content
        
        # Cache the result
        cache[cache_key] = res
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        
        print(f"[{question_number} new]")
        
        del model_inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        return res



    def generate_batch(self, records, cache, cache_file, user_prompt, system_prompt=None,
                       temperature=0.6, max_new_tokens=100):
        """
        
        Generate answer using the pre-loaded model given cache file to avoid reundant inference calls.
        
        Args:
        user_inputs: dict of inputs : eg. {'question': "", 'reference': ""}
        question number: ID or idx number
        cache: current cache dict
        cache_file: path to cache file
        user_prompt: Prompt template, role: user
        system_prompt: Prompt template, role:system
        temperature: generation temperature -> default=0.6
        max_new_tokens: default 100

        """
        texts, uncached_records, cache_keys = [], [], []

        for record in records:
            cache_key = f"{self.model_name}::{record['question']}"
            cache_keys.append(cache_key)

            if cache_key not in cache:
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt.format(**record)}
                    ]
                else:
                    messages = [{"role": "user", "content": user_prompt.format(**record)}]

                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
                uncached_records.append(record)

        if not texts:  # everything cached
            return cache

        # Batch tokenize
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,   #
                use_cache=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode per record
        for idx, record in enumerate(uncached_records):
            output_ids = generated_ids[idx][len(model_inputs.input_ids[idx]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            res = self.extract_answer_tags(content)
            cache_key = f"{self.model_name}::{record['question']}"
            cache[cache_key] = {
                "reasoning": content,   # full decoded text (reasoning + answer)
                "score": res            # extracted <answer>…</answer>
            }

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        del model_inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()
        return cache
        
    def regenerate_resp(self, record, num_tries, user_prompt_template, max_new_tokens=2048, 
                   temperature=0.01):
        """
        record: dict: {question, reference, answer}
        num_tries : max number of tries
        """
    
        messages = [{"role": "user", "content": user_prompt_template.format(**record)}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer(
        text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        try_i = 0
        resp_len = 10
        while try_i < num_tries and resp_len > 1:
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,   #
                    use_cache=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            res = self.extract_answer_tags(content)
            resp_len = len(res)
            try_i += 1

        record['score'] = res
        print(record['score'])
            
        return record



def process_questions_hf(question_df, df_type, prompt_template, model_name):
    cache_file = f"gpqa_diamond_cache_{df_type}_{model_name}_answers.json"
    # Load cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    if "qwen" in model_name.lower():
        qwen_inference = HFInference(f"Qwen/{model_name}")
    
    results = []
    
    # Process all questions with the same model instance
    for i, record in enumerate(question_df):
        answer = qwen_inference.answer_question(
            record, 
            i + 1, 
            cache, 
            cache_file, 
            prompt_template
        )
        results.append({
            'question': record['question'],
            'answer': answer
        })
        
    answer_df = pd.DataFrame(results, columns=["question", "answer"])
    print("Processing complete.")
    del qwen_inference.model 
    del qwen_inference.tokenizer 
    del qwen_inference 
    gc.collect() 
    torch.cuda.empty_cache() 
    return answer_df

def process_batch(mod_inference, batch, cache, cache_file, user_prompt, system_prompt, temperature, max_new_tokens, batch_id=None, worker_id=None):
    cache = mod_inference.generate_batch(
        batch, cache, cache_file, user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    results = []
    for record in batch:
        cache_key = f"{mod_inference.model_name}::{record['question']}"
        results.append({
            'judge': mod_inference.model_name,
            'question': record['question'],
            'reference': record['reference'],
            'response': record['answer'],
            'score': cache[cache_key]['score']
        })


    print(f" Processed batch {batch_id} ({len(batch)} records)")

    return results, cache

def run_inference(responses_df, model, cache, cache_name, user_prompt, system_prompt,
                  temperature, max_new_tokens, batch_size=4, split_data=True, split_model=True):
    """
    responses_df: list of dicts : dataset records
    model: str : model name
    cache: dict : existing cache
    cache_name: str : name identifier for cache file
    user_prompt: str : prompt for generation
    system_prompt: str : optional system prompt
    split_data: bool : True to split data and create multiple instances if possible - 1 instance per gpu
    split_model: bool : True to split model across multiple GPUs if possible
    """
    results = []
    cache_file = f"gpqa_diamond_cache_{cache_name}_{model}_matches.json"

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    if "qwen" in model.lower() and num_gpus > 0:

        if split_model:
            # Decide number of instances
            # For example: if num_gpus = 4 -> 2 instances of 2 GPUs each
            # if num_gpus = 6 -> 2 instances of 3 GPUs each or 3 instances of 2 GPUs
            # Let's split as evenly as possible into 2 instances if num_gpus is even, else 2 or 3
            num_instances = 1
            if num_gpus >= 4 and num_gpus % 2 == 0:
                num_instances = num_gpus // 2 
            else:
                num_instances = 1  # fallback

            # Group GPUs
            group_size = num_gpus // num_instances
            gpu_groups = []
            for i in range(num_instances):
                start = i * group_size
                end = start + group_size
                if i == num_instances - 1:
                    end = num_gpus  # include remaining
                gpu_groups.append(list(range(start, end)))
            print(f"Splitting into {num_instances} instance(s) with GPU groups: {gpu_groups}")

            # Split data into num_instances parts
            splits = []
            split_size = math.ceil(len(responses_df) / num_instances)
            for i in range(num_instances):
                start = i * split_size
                end = min((i + 1) * split_size, len(responses_df))
                splits.append(responses_df[start:end])

            # Prepare model instances per GPU group
            mod_inferences = []
            for i, gpu_group in enumerate(gpu_groups):
                if len(gpu_group) == 1:
                    device = f"cuda:{gpu_group[0]}"
                    print(f"Loading instance {i} on {device}")
                    if "qwen" in model.lower():
                        mod = HFInference(f"Qwen/{model}", device)
                else:
                    # Model parallelism across multiple GPUs
                    print(f"Loading instance {i} across GPUs {gpu_group}")
                    max_memory = {}
                    reserve_per_gpu_gb = 2
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        total_gb = props.total_memory / (1024 ** 3)  # Convert bytes to GB
                        usable_gb = max(total_gb - reserve_per_gpu_gb, 1)  # Ensure at least 1GB remains
                        max_memory[i] = f"{int(usable_gb)}GB"
                        
                    if "qwen" in model.lower():
                        # max_memory = {gpu: "24GB" for gpu in gpu_group}  
                        mod = HFInference(f"Qwen/{model}", device_map="auto", max_memory=max_memory)

                mod_inferences.append(mod)

            # Prepare batches
            batches_list = []
            for i in range(num_instances):
                split = splits[i]
                batches = [split[j:j + batch_size] for j in range(0, len(split), batch_size)]
                batches_list.append(batches)

            # Run inference in parallel
            with ThreadPoolExecutor(max_workers=num_instances) as executor:
                futures = []
                for i in range(num_instances):
                    mod = mod_inferences[i]
                    for batch_id, batch in enumerate(batches_list[i], start=1):
                        futures.append(executor.submit(
                            process_batch, mod, batch, cache, cache_file, user_prompt,
                            system_prompt, temperature, max_new_tokens,
                            batch_id=batch_id, worker_id=i
                        ))

                for f in as_completed(futures):
                    batch_results, cache = f.result()
                    results.extend(batch_results)

            # Cleanup
            for mod in mod_inferences:
                del mod.model, mod.tokenizer, mod

        elif split_data:
            # Data parallelism only: one instance per GPU, proc split over multiple gpus
            if "qwen" in model.lower():
                mod_inferences = [
                    HFInference(f"Qwen/{model}", f"cuda:{i}") for i in range(num_gpus)
                ]

            # Split data into num_gpus parts
            splits = []
            split_size = math.ceil(len(responses_df) / num_gpus)
            for i in range(num_gpus):
                start = i * split_size
                end = min((i + 1) * split_size, len(responses_df))
                splits.append(responses_df[start:end])

            batches_list = [
                [splits[i][j:j + batch_size] for j in range(0, len(splits[i]), batch_size)]
                for i in range(num_gpus)
            ]

            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for gpu_id in range(num_gpus):
                    mod = mod_inferences[gpu_id]
                    for batch_id, batch in enumerate(batches_list[gpu_id], start=1):
                        futures.append(executor.submit(
                            process_batch, mod, batch, cache, cache_file, user_prompt,
                            system_prompt, temperature, max_new_tokens,
                            batch_id=batch_id, worker_id=gpu_id
                        ))

                for f in as_completed(futures):
                    batch_results, cache = f.result()
                    results.extend(batch_results)

            # Cleanup
            for mod in mod_inferences:
                del mod.model, mod.tokenizer, mod

        else:
            # Single instance on one GPU
            device = f"cuda:0"
            print(f"Using single GPU: {device}")
            if "qwen" in model.lower():
                mod_inference = HFInference(f"Qwen/{model}", device)
            for start in range(0, len(responses_df), batch_size):
                batch = responses_df[start:start + batch_size]
                batch_results, cache = process_batch(mod_inference, batch, cache, cache_file, user_prompt, system_prompt, temperature, max_new_tokens)
                results.extend(batch_results)
            del mod_inference.model, mod_inference.tokenizer, mod_inference

    else:
        # CPU fallback
        device = "cpu"
        print("Using CPU")
        mod_inference = HFInference(model, device)
        for start in range(0, len(responses_df), batch_size):
            batch = responses_df[start:start + batch_size]
            batch_results, cache = process_batch(mod_inference, batch, cache, cache_file, user_prompt, system_prompt, temperature, max_new_tokens)
            results.extend(batch_results)
        del mod_inference.model, mod_inference.tokenizer, mod_inference

    gc.collect()
    torch.cuda.empty_cache()
    print("Processing complete.")
    return pd.DataFrame(results, columns=["judge", "question", "reference", "response", "score"])

def regenerate(path, num_tries, user_prompt_template, model_name, max_new_tokens=2048, temperature=0.01):
    """
    Clean up any corrupted scoring - 
    ie, cases where the matcher was not able to finish reasoning to get to scoring

    record: dict: {question, reference, answer}
    num_tries : max number of tries

    returns the corrected merged df
    """ 
    df = pd.read_csv(path)
    
    score_str = df["score"].astype(str)
    invalid = df[score_str.str.len() > 2]
    invalid =  invalid.rename(columns={"response": "answer"})
    records = invalid.to_dict(orient="records")
    

    if "qwen" in model_name.lower():
        mod_inference = HFInference(f"Qwen/{model_name}")

    corrected = []
    print(len(records))
    for i, record in enumerate(records):
        # print(record)
        record = mod_inference.regenerate_resp(record, num_tries, user_prompt_template,
                                                max_new_tokens, temperature)
        corrected.append(record)

    corr_df = pd.DataFrame(corrected)

    base_path = path.split("/")[-1].split(".")[0] 
    valid_path = base_path + '_valid' + ".csv"
    # corr_df.to_csv(valid_path)
    print("Corrected!")
    old_df = df
    
    key = 'question'
    merged = old_df.set_index(key)
    merged.update(corr_df.set_index(key))
    
    merged = merged.reset_index()
    merged["flag_regen"] = merged[key].isin(corr_df[key])
    if 'merged' not in base_path.lower():
        merged.to_csv(base_path + '_merged.csv', index=False)
    else:
        merged.to_csv(base_path, index=False)
    print("Merged!")
    
# paths = ["/kaggle/input/gpqa-diamond/gpqa_diamond_quant_qwen_strategic_matches_baseline.csv", 
#          "/kaggle/input/gpqa-diamond/gpqa_diamond_qual_qwen_strategic_matches_baseline.csv", 
#          "/kaggle/input/gpqa-diamond/gpqa_diamond_quant_gpt_strategic_matches_baseline.csv", 
#          "/kaggle/input/gpqa-diamond/gpqa_diamond_qual_gpt_strategic_matches_baseline.csv", 
#         ]
# for p in paths:
#     regenerate(p, 3, user_prompt_template, "Qwen3-4B")

def extract_reference(row):
    try:
        # split on the second character of options string
        options_list = [
            opt.strip() 
            for opt in row["options"].split(row["options"][1]) 
            if opt.strip() and opt.strip() not in ['[', ']']
        ]
        return options_list[int(row["answer_index"])]
    except (IndexError, ValueError, TypeError):
        # fallback for invalid rows
        return None

def get_resp_df(q_df, a_df, bench):
    if bench == "gpqa":
        resp = q_df[['question', 'reference', 'question_mcq']].merge(a_df, on='question')
        resp = resp[['question', 'reference', 'answer']]
    if bench == "mmlu":
        q_df["reference"] = q_df.apply(extract_reference, axis=1)
        resp = q_df[['question', 'reference']].merge(a_df, on='question')
        resp = resp[['question', 'reference', 'answer']]
    return resp


def main():
    user_prompt_template = answer_matching_prompts.get_judge_prompt_with_gt_baseline() 


    parser = argparse.ArgumentParser()

    parser.add_argument('--testee_name', default="Qwen2.5-7B-Instruct", type=str)
    parser.add_argument('--matcher_name', default="Qwen3-4B", type=str)
    parser.add_argument('--exp', default="baseline", type=str)

    #process only so many data points
    parser.add_argument('--max_sample_size', default=5, type=int)

    parser.add_argument('--testee_prompt', default=answer_matching_prompts.BASELINE_PROMPT, type=str)
    parser.add_argument('--matcher_prompt', default=user_prompt_template, type=str)

    parser.add_argument('--temperature_testee', default=0.6, type=float)
    parser.add_argument('--max_new_tokens_testee', default=300, type=int)
    parser.add_argument('--temperature_matcher', default=0.01, type=float)
    parser.add_argument('--max_new_tokens_matcher', default=2048, type=int)

    parser.add_argument('--cache_name', default="qwen7b", type=str)
    parser.add_argument('--num_retries', default=3, type=int)

    #matcher
    parser.add_argument('--model_split', default=False, type=bool)
    parser.add_argument('--data_split', default=True, type=bool)


    parser.add_argument('--base_qual_data', default="/kaggle/input/gpqa-diamond/gpqa_diamond_qualitative.csv", type=str)
    parser.add_argument('--base_quant_data', default="/kaggle/input/gpqa-diamond/gpqa_diamond_quantitative.csv", type=str)

    #paths to read from - matcher
    parser.add_argument('--qual_ans', default="/kaggle/input/gpqa-diamond/gpqa_diamond_qual_qwen_answers.csv", type=str)
    parser.add_argument('--quant_ans', default="/kaggle/input/gpqa-diamond/gpqa_diamond_quant_qwen_answers.csv", type=str)

    #paths to save to
    parser.add_argument('--qual_gen_op', default="qwen_qual_baseline_answers.csv", type=str)
    parser.add_argument('--quant_gen_op', default="qwen_quant_baseline_answers.csv", type=str)
    parser.add_argument('--qual_match_op', default="qwen_qual_baseline_matches.csv", type=str)
    parser.add_argument('--quant_match_op', default="qwen_quant_baseline_matches.csv", type=str)

    #==========================================================================================#
    
    args, unknown = parser.parse_known_args()

    args.testee_name = "Qwen2.5-7B-Instruct"
    args.matcher_name = "Qwen3-4B"
    args.testee_prompt = answer_matching_prompts.BASELINE_PROMPT
    args.matcher_prompt = user_prompt_template
    args.cache_name = "qwen7b"
    args.model_split = False
    args.data_split = True
    args.exp = "baseline"
    args.max_sample_size = 5000000 #whole dataset

    # args.qual_gen_op = "qwen_qual_verbose_answers.csv"
    # args.quant_gen_op = "qwen_quant_verbose_answers.csv"
    # args.qual_ans = "qwen_qual_verbose_answers.csv"
    # args.quant_ans = "qwen_quant_verbose_answers.csv"

    # args.qual_match_op = "qwen_qual_verbose_matches.csv"
    # args.quant_match_op = "qwen_quant_verbose_matches.csv"


    args.qual_gen_op = f"gpqa/{args.exp}/qwen_qual_baseline_answers.csv"
    args.quant_gen_op = f"gpqa/{args.exp}/qwen_qual_baseline_answers.csv"
    args.qual_ans = f"gpqa/{args.exp}/qwen_qual_baseline_answers.csv"
    args.quant_ans = f"gpqa/{args.exp}/qwen_quant_baseline_answers.csv"

    args.qual_match_op = f"scores/gpqa/{args.exp}/{args.matcher_name}/qwen_qual_baseline_matches.csv"
    args.quant_match_op = f"scores/gpqa/{args.exp}/{args.matcher_name}/qwen_quant_baseline_matches.csv"

    qual = pd.read_csv(args.base_qual_data)
    quant = pd.read_csv(args.base_quant_data)

    #testee--generate
    qual_data = random_samples(qual.to_dict(orient="records"), args.max_sample_size)
    quant_data = random_samples(quant.to_dict(orient="records"), args.max_sample_size)

    qwen_qual_answer_df = process_questions_hf(qual_data, args.cache_name, 
                                            args.testee_prompt, args.testee_name)
    qwen_qual_answer_df.to_csv(args.qual_gen_op, index=False)

    qwen_quant_answer_df = process_questions_hf(quant_data, args.cache_name, 
                                                args.testee_prompt, args.testee_name)
    qwen_quant_answer_df.to_csv(args.quant_gen_op, index=False)


    #judge--match
    qwen_responses = pd.read_csv(args.qual_ans)
    qual_responses_qwen = get_resp_df(qual, qwen_responses, "gpqa")
    qwen_responses = pd.read_csv(args.quant_ans)
    quant_responses_qwen = get_resp_df(quant, qwen_responses, "gpqa")

    print(f"Judging {args.qual_ans}")
    qwen_qual_am_df = run_inference(responses_df=qual_responses_qwen.to_dict(orient="records"),
                                    model=args.matcher_name, 
                                    cache={}, 
                                    cache_name=args.cache_name, 
                                    user_prompt=args.matcher_prompt, 
                                    system_prompt=None,
                                    temperature=args.temperature_matcher, 
                                    max_new_tokens=args.max_new_tokens_matcher, 
                                    split_data=args.model_split, 
                                    split_model=args.data_split)
        
    qwen_qual_am_df.to_csv(args.qual_match_op, index=False)

    print(f"Judging {args.quant_ans}")
    qwen_quant_am_df = run_inference(responses_df=quant_responses_qwen.to_dict(orient="records"),
                                    model=args.matcher_name, 
                                    cache={}, 
                                    cache_name=args.cache_name, 
                                    user_prompt=args.matcher_prompt, 
                                    system_prompt=None,
                                    temperature=args.temperature_matcher, 
                                    max_new_tokens=args.max_new_tokens_matcher, 
                                    split_data=args.model_split, 
                                    split_model=args.data_split)
        
    qwen_quant_am_df.to_csv(args.quant_match_op, index=False)
    
if __name__ == "__main__":
    main()
