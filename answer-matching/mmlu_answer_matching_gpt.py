#!/usr/bin/env python
# coding: utf-8

import os
import json
import re
import gc
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
# from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import Optional
import torch

from answer_matching_prompts import (
    JUDGE_PROMPT_TEMPLATE_MMLU,
    JUDGE_PROMPT_TEMPLATE_MMLU_CONT,
    JUDGE_PROMPT_TEMPLATE_GEMMA_BINARY,
    JUDGE_PROMPT_TEMPLATE_GEMMA_CONT,
    JUDGE_PROMPT_TEMPLATE_TRAD_BINARY,
    JUDGE_PROMPT_TEMPLATE_TRAD_CONT
)

# Load API key
client = OpenAI(api_key="")

cache_lock = threading.Lock()


def extract_numeric_score(response_text: str) -> float:
    """
    Extract 0/1 score from <answer> tags in the model response.
    Falls back to 0.0 if parsing fails.
    """
    matches = re.findall(
        r"<answer>\s*([01](?:\.\d+)?)\s*</answer>", 
        response_text, re.IGNORECASE | re.DOTALL
    )
    if matches:
        val = float(matches[0])
        return max(0.0, min(1.0, val))  # clamp to [0,1]
    return 0.0


def judge_batch(batch, cache, cache_file, user_prompt, system_prompt="", temperature=0.0, max_new_tokens=200):
    """
    Send a batch of question-answer-reference triples to OpenAI API for judgment.
    """
    results = []
    for record in batch:
        cache_key = f"gpt-judge::{record['question']}::{record['answer']}"
        with cache_lock:
            if cache_key in cache:
                response_text = cache[cache_key]
            else:
                prompt_text = user_prompt.format(
                    question=record["question"],
                    # reference=record["reference"],
                    answer=record["answer"]
                )

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_output_tokens=max_new_tokens,
                    temperature=temperature
                )
                response_text = response.output_text.strip()

                cache[cache_key] = response_text
                with open(cache_file, "w") as f:
                    json.dump(cache, f, indent=2)

        score = extract_numeric_score(response_text)
        results.append({
            "judge": "gpt-4.1-mini",
            "question": record["question"],
            # "reference": record["reference"],
            "answer": record["answer"],
            "score": score
        })
    return results


def answer_match(
    responses_df,
    answer_df,
    response_type,
    q_type,
    user_prompt,
    system_prompt="",
    temperature=0.0,
    max_new_tokens=200,
    batch_size=4,
):
    """
    Compare model-generated answers against ground-truth answers using OpenAI judge model.
    """

    cache_file = f"trad_gpt_{response_type}_{q_type}_cont_answer_matches.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Convert responses
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
                # "reference": ref_answer,
                "answer": response_row.get("answer", "")
            }
            batch.append(record_data)

        if not batch:
            continue

        batch_results = judge_batch(
            batch, cache, cache_file, user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        for r in batch_results:
            results.append(r)
            print(f"{question_num}: {r['score']}")
            question_num += 1

    df_out = pd.DataFrame(results, columns=["judge", "question", "reference", "answer", "score"])

    gc.collect()
    torch.cuda.empty_cache()
    print("Processing complete.")
    return df_out


def main():
    answers_df = pd.read_csv("answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_quant_answers_final_sample.csv")
    base_data = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_final_sample.csv")

    matched_df = answer_match(
        responses_df=answers_df,
        answer_df=base_data,
        response_type="verbose",
        q_type="quant_cont",
        user_prompt=JUDGE_PROMPT_TEMPLATE_TRAD_CONT,
        system_prompt="",
        temperature=0.0,
        max_new_tokens=200,
        batch_size=10
    )
    
    matched_df.to_csv("trad_gpt_mmlu_pro_verbose_quant_cont_GPT-4.1-mini_judge.csv", index=False)


if __name__ == "__main__":
    main()
