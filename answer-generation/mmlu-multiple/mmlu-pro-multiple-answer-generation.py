#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from answer_generation_prompts import MMLU_FORWARD_PROMPT, MMLU_BACKWARD_PROMPT
import threading

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cache_lock = threading.Lock()

MAX_WORKERS = 8

def answer_question(question_text, choices, question_number, category, cache, cache_file, PROMPT_TEMPLATE):
    cache_key = f"{category}::{question_text}"
    with cache_lock:
        if cache_key in cache:
            # Just a small, single-line progress print
            print(f"[{question_number} cached]")
            return cache[cache_key]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=PROMPT_TEMPLATE.format(question=question_text,
                                           choices=", ".join(choices)),
        max_output_tokens=300,
        temperature=0
    )

    text_response = response.output_text.strip()

    with cache_lock:
        cache[cache_key] = text_response
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"[{question_number} new]")

    return text_response


def generate_answers(question_df, df_type, PROMPT_TEMPLATE, prompt_type):
    cache_file = f"gpt_mmlu_pro_{prompt_type}_cache_{df_type}_answers.json"

    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Collect questions to process (skip ones already cached)
    to_process = [(i, ex) for i, ex in enumerate(question_df)
                  if f"{ex['category']}::{ex['question']}" not in cache]

    total = len(to_process)
    progress = [0]  # list so it can be mutated inside threads

    # For storing answers in a dict keyed by index to preserve order
    indexed_results = {}

    def process_item(item):
        idx, example = item
        q_text = example['question']
        choices = example['options']
        cat = example['category']
        ans = answer_question(q_text, choices, idx, cat, cache, cache_file, PROMPT_TEMPLATE)
        with cache_lock:
            progress[0] += 1
            print(f"✓ Answered question {progress[0]}/{total}")
            cache[f"{cat}::{q_text}"] = ans
            indexed_results[idx] = (cat, q_text, ans)
        return ans

    # Process in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_item, item): item for item in to_process}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                idx, _ = futures[future]
                print(f"Error processing question {idx}: {e}")

    # Add already cached items to results
    for i, ex in enumerate(question_df):
        key = f"{ex['category']}::{ex['question']}"
        if key in cache:
            indexed_results[i] = (ex['category'], ex['question'], cache[key])

    # Save updated cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Sort by original question order
    sorted_results = [indexed_results[i] for i in sorted(indexed_results)]
    answer_df = pd.DataFrame(sorted_results, columns=["category", "question", "answer"])
    print("Processing complete.")
    return answer_df

def main():
    qual_test = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test.csv")
    quant_test = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_test.csv")

    qual_valid = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_validation.csv")
    quant_valid = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_validation.csv")

    prompt_template_list = [MMLU_BACKWARD_PROMPT]
    prompt_types_list = ["backward"]

    for i in range(len(prompt_template_list)):
        prompt_temp = prompt_template_list[i]
        prompt_type = prompt_types_list[i]

        qual_test_answer_df = generate_answers(qual_test.to_dict(orient="records"), "qual_test", prompt_temp, prompt_type)
        quant_test_answer_df = generate_answers(quant_test.to_dict(orient="records"), "quant_test", prompt_temp, prompt_type)
        qual_valid_answer_df = generate_answers(qual_valid.to_dict(orient="records"), "qual_valid", prompt_temp, prompt_type)
        quant_valid_answer_df = generate_answers(quant_valid.to_dict(orient="records"), "quant_valid", prompt_temp, prompt_type)

        qual_test_answer_df.to_csv(f"gpt_mmlu_pro_{prompt_type}_qual_test_answers.csv", index=False)
        quant_test_answer_df.to_csv(f"gpt_mmlu_pro_{prompt_type}_quant_test_answers.csv", index=False)
        qual_valid_answer_df.to_csv(f"gpt_mmlu_pro_{prompt_type}_qual_valid_answers.csv", index=False)
        quant_valid_answer_df.to_csv(f"gpt_mmlu_pro_{prompt_type}_quant_valid_answers.csv", index=False)

if __name__ == "__main__":
    main()