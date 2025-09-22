import answer_matching_prompts
from inference import HFInference
import os
import json
import re
import ast
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
import torch, gc
from dataclasses import dataclass
from typing import Optional
from answer_matching_prompts import JUDGE_PROMPT_TEMPLATE_MMLU, JUDGE_PROMPT_TEMPLATE_MMLU_CONT
import random

HF_CACHE_DIR = "/workspace/Gaming-the-Answer-Matcher/hf_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR

def answer_match(
    responses_df,
    answer_df,
    model,
    response_type,
    q_type,
    user_prompt,
    system_prompt=None,
    temperature=0.6,
    max_new_tokens=100,
    batch_size=4,
):
    """
    Prompt for model judgement given question, answers and/or references.

    responses_df: DataFrame or list[dict] with columns/keys: question, reference, answer
    answer_df:    DataFrame with columns/keys: question, reference, options
    model:        e.g. "Qwen3-4B"
    user_prompt:  prompt template string
    """

    # init inference backend
    if "qwen" in model.lower():
        mod_inference = HFInference(f"Qwen/{model}")
        # mod_inference = HFInference("/workspace/hf_cache/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796")
    else:
        mod_inference = HFInference(model)

    cache_file = f"{model}_{response_type}_{q_type}_answer_matches.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # convert responses to dict records
    if isinstance(responses_df, pd.DataFrame):
        responses_records = responses_df.to_dict(orient="records")
    else:
        responses_records = responses_df

    answer_records = answer_df.to_dict(orient="records")
    results = []
    total_batches = (len(answer_records) + batch_size - 1) // batch_size

    question_num = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = []

        for record in answer_records[start:end]:
            question = record["question"]
            # merge with model responses for this question
            response_row = next((r for r in responses_records if r["question"] == question), {})
            options_list = [opt.strip() for opt in record["options"].split(record["options"][1]) if opt.strip() and opt.strip() not in ['[', ']']]
            try:
                ref_answer = options_list[int(record["answer_index"])]
            except (IndexError, ValueError):
                # Skip this question if answer_index is invalid
                print(f"Skipping question due to invalid index: {question}")
                continue
            record_data = {
                "question": question,
                "reference": options_list[int(record["answer_index"])],
                "answer": response_row.get("answer", "")
            }
            batch.append(record_data)

        # skip empty batch
        if not batch:
            continue

        # run inference on batch
        cache = mod_inference.generate_batch(
            batch, cache, cache_file, user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        # collect results
        for record in batch:
            cache_key = f"{mod_inference.model_name}::{record['question']}"
            score_val = cache[cache_key]["score"]
            results.append({
                "judge": model,
                "question": record["question"],
                "reference": record["reference"],
                "answer": record["answer"],
                "score": score_val
            })
            print(f"{question_num}: {score_val}")
            question_num += 1

    # convert results to DataFrame
    answer_df_out = pd.DataFrame(results, columns=["judge", "question", "reference", "answer", "score"])

    # cleanup
    del mod_inference.model
    del mod_inference.tokenizer
    del mod_inference
    gc.collect()
    torch.cuda.empty_cache()

    print("Processing complete.")
    return answer_df_out

    # cleanup VRAM 
    del mod_inference.model
    del mod_inference.tokenizer
    del mod_inference
    gc.collect()
    torch.cuda.empty_cache()
    print("Processing complete.")
    return answer_df

def main():
    answers_df = pd.read_csv("answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_qual_test_answers.csv")
    base_data = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test_sample_large.csv")

    matched_df = answer_match(
        responses_df = answers_df,
        answer_df = base_data,
        model = "Qwen3-4B",
        response_type = "verbose",
        q_type = "qual",
        user_prompt = JUDGE_PROMPT_TEMPLATE_MMLU,
        system_prompt = None,
        temperature = 0.0,
        max_new_tokens = 256,
        batch_size = 10
    )
    matched_df.to_csv("gpt_mmlu_pro_verbose_qual_test_Qwen3-4B_judge.csv")
    
if __name__ == "__main__":
    main()
