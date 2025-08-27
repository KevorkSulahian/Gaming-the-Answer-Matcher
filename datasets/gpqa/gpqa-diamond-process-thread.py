from datasets import load_dataset
import os
import json
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cache_lock = threading.Lock()

MAX_WORKERS = 8

def classify_question(question_text, question_number, cache, cache_file):
    cache_key = f"GPQA::{question_text}"
    with cache_lock:
        if cache_key in cache:
            reasoning_type = cache[cache_key]
            return reasoning_type

    prompt = f"""
    You are labeling multiple-choice exam questions as either quantitative or qualitative.

    - A **quantitative question** requires calculation, numeric reasoning, or solving equations.
    - A **qualitative question** requires conceptual understanding, factual recall, or reasoning without computation.

    Question:
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

    return reasoning_type

def process_split():
    dataset = pd.read_parquet("gpqa_diamond.parquet")
    cache_file = f"gpqa_diamond_reasoning_cache.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    to_process = [
        (i, row)
        for i, row in dataset.iterrows()
        if f"GPQA::{row['question']}" not in cache
    ]

    print(f"Processing {len(to_process)} questions in parallel...")

    reasoning_types = [cache.get(f"GPQA::{row['question']}", None) for _, row in dataset.iterrows()]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_question, row["question"], i, cache, cache_file): i
            for i, row in to_process
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="progress"):
            i = futures[future]
            reasoning_types[i] = future.result()

    dataset["reasoning_type"] = reasoning_types

    quant_dataset = dataset[dataset["reasoning_type"] == "quantitative"].drop(columns=["reasoning_type"])
    qual_dataset = dataset[dataset["reasoning_type"] == "qualitative"].drop(columns=["reasoning_type"])

    return quant_dataset, qual_dataset

def clean_mcq():
    def extract_reference(text, ans):
        lines = text.splitlines()
        for line in lines:
            if f"{ans}." == line[:2]:
                return line[2:]
        print(lines)
    def extract_choices(text):
        choices = re.findall(r"[A-D]\.\s*(.+)", text)
        choices_str = ", ".join(choices)
        return choices_str

    qual_path = "../../datasets/gpqa/gpqa_diamond_qualitative.csv"
    quant_path = "../../datasets/gpqa/gpqa_diamond_quantitative.csv"
    qual = pd.read_csv(qual_path)
    quant = pd.read_csv(quant_path)

    qual['reference'] = qual.apply(lambda row: extract_reference(row['question'], row['answer']), axis=1)
    qual['question_mcq'] = qual['question']
    qual['question'] = qual['question'].apply(lambda x: '\n'.join(x.splitlines()[:-4]))
    qual['choices'] = qual.apply(lambda row: extract_choices(row['question_mcq']), axis=1)
    qual.to_csv(qual_path)

    quant['reference'] = quant.apply(lambda row: extract_reference(row['question'], row['answer']), axis=1)
    quant['question_mcq'] = quant['question']
    quant['question'] = quant['question'].apply(lambda x: '\n'.join(x.splitlines()[:-4]))
    
    quant['choices'] = quant.apply(lambda row: extract_choices(row['question_mcq']), axis=1)

    quant.to_csv(quant_path)

    

def main():
    quant_data, qual_data = process_split()

    quant_data.to_csv(f"gpqa_diamond_quantitative.csv", index=False)
    qual_data.to_csv(f"gpqa_diamond_qualitative.csv", index=False)
    # clean_mcq()

if __name__ == "__main__":
    main()
