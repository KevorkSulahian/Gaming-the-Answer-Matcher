import answer_matching_prompts
from inference import HFInference
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
from dataclasses import dataclass
from typing import Optional
from answer_matching_prompts import JUDGE_PROMPT_TEMPLATE_WITH_GT_baseline
import random

def answer_match(
    responses_df,
    answer_df,
    model,
    user_prompt,
    system_prompt=None,
    temperature=0.6,
    max_new_tokens=100,
    batch_size=4,
    surface_mode: str = "medium",   # "light" | "medium" | "heavy"
    surface_seed: Optional[int] = None,
):
    """
    Prompt for model judgement given question, answers and/or references.

    responses_df: list[dict] or DataFrame with columns/keys: question, reference, answer
    df_type:      'qual' | 'quant' | etc.
    model:        e.g. "Qwen3-4B"
    user_prompt:  prompt template string
    """

    # init inference backend
    if "qwen" in model.lower():
        mod_inference = HFInference(f"Qwen/{model}")
    else:
        mod_inference = HFInference(model)

    cache_file = f"{model}_{responses_df}_answer_matches.json"

    if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
    else:
        cache = {}

    responses_df = responses_df.reset_index(drop=True)
    answer_df = answer_df.reset_index(drop=True)
    score_df = pd.DataFrame(columns=["question", "reference", "answer", "score"])

    for index, row in responses_df.iterrows():
        question = row['question']
        reference = answer_df.iloc[index]['reference']
        answer = row['answer']

        record = {
            'question': question,
            'reference': reference,
            'answer': answer
        }

        cache_key = f"{mod_inference.model_name}::{question}"

        if cache_key in cache:
            print(f"Cache hit for question: {question}")
            continue

        score_response = mod_inference.generate(
            prompt=JUDGE_PROMPT_TEMPLATE_WITH_GT_baseline.format(**record),
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        score_df.loc[len(score_df)] = {
            "question": question,
            "reference": reference,
            "answer": answer,
            "score": score_response
        }

        cache[cache_key] = score_response
        with open(cache_file, "w") as f:
            json.dump(cache, f)

        print(f"Processed question: {question}")

    

    # results = []
    # b_count = 0

    # for start in range(0, len(records), batch_size):
    #     original_batch = records[start:start + batch_size]

    #     # make a transformed copy so we don't mutate the caller's data
    #     batch = []
    #     for rec in original_batch:
    #         rec2 = dict(rec)  
    #         if use_surface_attack:
    #             # keep raw for audit; judge sees manipulated
    #             rec2["_answer_raw"] = rec2.get("answer", "")
    #             rec2["answer"] = surface_manipulate(rec2.get("answer", ""), cfg)
    #         else:
    #             rec2["_answer_raw"] = rec2.get("answer", "")
    #         batch.append(rec2)

    #     # run judge
    #     cache = mod_inference.generate_batch(
    #         batch, cache, cache_file, user_prompt,
    #         system_prompt=system_prompt,
    #         temperature=temperature,
    #         max_new_tokens=max_new_tokens
    #     )

    #     # collect results
    #     for record in batch:
    #         cache_key = f"{mod_inference.model_name}::{record['question']}"
    #         score_val = cache[cache_key]['score']
    #         results.append({
    #             'judge': model,
    #             'df_type': df_type,
    #             'attack': attack_tag_col,         
    #             'question': record['question'],
    #             'reference': record['reference'],
    #             'response_raw': record['_answer_raw'], 
    #             'response': record['answer'],          
    #             'score': score_val
    #         })
    #         print(score_val)

    #     b_count += 1
    #     print(f"Proc {b_count} batch")

    # answer_df = pd.DataFrame(
    #     results,
    #     columns=[
    #         "judge", "df_type", "attack",
    #         "question", "reference", "response_raw", "response", "score"
    #     ]
    # )

    # cleanup VRAM 
    del mod_inference.model
    del mod_inference.tokenizer
    del mod_inference
    gc.collect()
    torch.cuda.empty_cache()
    print("Processing complete.")
    return answer_df

def main():
    gpt_mmlu_pro_forward_qual = pd.read_csv("answer-generation/mmlu-multiple/gpt_mmlu_pro_forward_qual_test_answers.csv")
    gpt_mmlu_pro_forward_quant = pd.read_csv("answer-generation/mmlu-multiple/gpt_mmlu_pro_forward_quant_test_answers.csv")

    gpt_mmlu_pro_backward_qual = pd.read_csv("answer-generation/mmlu-multiple/gpt_mmlu_pro_backward_qual_test_answers.csv")
    gpt_mmlu_pro_backward_quant = pd.read_csv("answer-generation/mmlu-multiple/gpt_mmlu_pro_backward_quant_test_answers.csv")

    mmlu_pro_qual_test = pd.read_csv("data/mmlu_pro_qual_test.csv")

    # qwen_gpqa_diamond_forward_qual = pd.read_csv("answer-generation/gpqa-multiple/qwen_gpqa_diamond_forward_qual_answers.csv")
    # qwen_gpqa_diamond_forward_quant = pd.read_csv("answer-generation/gpqa-multiple/qwen_gpqa_diamond_forward_quant_answers.csv")

    # qwen_gpqa_diamond_backward_qual = pd.read_csv("answer-generation/gpqa-multiple/qwen_gpqa_diamond_backward_qual_answers.csv")
    # qwen_gpqa_diamond_backward_quant = pd.read_csv("answer-generation/gpqa-multiple/qwen_gpqa_diamond_backward_quant_answers.csv")

    answer_match(
        responses_df = gpt_mmlu_pro_forward_qual,
        answer_df = mmlu_pro_qual_test,
        model = "Qwen3-4B",
        user_prompt = JUDGE_PROMPT_TEMPLATE_WITH_GT_baseline,
        system_prompt = None,
        temperature = 0.0,
        max_new_tokens = 256,
        batch_size = 4
    )

if __name__ == "__main__":
    main()
