from datasets import load_dataset
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cache_lock = threading.Lock() 

splits = ["validation", "test"] 

MAX_WORKERS = 8

def classify_question(question_text, subject, question_number, cache, cache_file):
    cache_key = f"{subject}::{question_text}"
    with cache_lock:
        if cache_key in cache:
            reasoning_type = cache[cache_key]
            print(f"[{question_number}] {reasoning_type}")
            return reasoning_type

    prompt = f"""
    You are labeling exam questions as either quantitative or qualitative.

    - A **quantitative question** requires calculation, numeric reasoning, or solving equations.
    - A **qualitative question** requires conceptual understanding, factual recall, or reasoning without computation.

    Question (subject: {subject}):
    {question_text}

    Answer ONLY with 'quantitative' or 'qualitative'.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=16,
        temperature=0
    )

    reasoning_type = response.output_text.strip().lower()

    with cache_lock:
        cache[cache_key] = reasoning_type
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        print(f"[{question_number}] {reasoning_type}")

    return reasoning_type

def process_split(split_name):
    print(f"\nProcessing split: {split_name}")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split_name)
    cache_file = f"mmlu_pro_reasoning_cache_{split_name}.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    to_process = [(i, ex) for i, ex in enumerate(dataset)
                  if f"{ex['category']}::{ex['question']}" not in cache]

    print(f"Processing {len(to_process)} questions in parallel...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_question, ex["question"], ex["category"], i, cache, cache_file): i
                   for i, ex in to_process}

        for future in as_completed(futures):
            _ = future.result() 

    def attach_from_cache(example):
        key = f"{example['category']}::{example['question']}"
        example["reasoning_type"] = cache.get(key, "unknown")
        return example

    labeled_dataset = dataset.map(attach_from_cache)

    labeled_dataset.save_to_disk(f"mmlu_pro_labeled_{split_name}")

    quant_dataset = labeled_dataset.filter(lambda x: x["reasoning_type"] == "quantitative")
    qual_dataset = labeled_dataset.filter(lambda x: x["reasoning_type"] == "qualitative")

    return quant_dataset, qual_dataset

def main():
    for split in splits:
        quant_data, qual_data = process_split(split)
        quant_data = quant_data.to_pandas().drop(columns=["reasoning_type"])
        qual_data = qual_data.to_pandas().drop(columns=["reasoning_type"])
        quant_data.to_csv(f"mmlu_pro_quantitative_{split}.csv", index=False)
        qual_data.to_csv(f"mmlu_pro_qualitative_{split}.csv", index=False)

if __name__ == "__main__":
    main()
