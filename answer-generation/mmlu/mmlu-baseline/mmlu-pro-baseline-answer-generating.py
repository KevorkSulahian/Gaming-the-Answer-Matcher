#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import asyncio
import argparse
import logging
import re
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
import threading
import torch, gc

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
hf_client = InferenceClient() #ensure hf_token w/ finegrained inference permission is defined
cache_lock = threading.Lock()

MAX_WORKERS = 8

QUERY_PROMPT_TEMPLATE = """
You will be asked a question. Please provide your answer as a free-text response that is 3-4 sentences long. Keep your answer concise and to the point. Do not include any additional information or context beyond what is necessary to answer the question.

Question: {question}
"""

def answer_question(question_text, question_number, category, cache, cache_file):
    cache_key = f"{category}::{question_text}"
    with cache_lock:
        if cache_key in cache:
            # Just a small, single-line progress print
            print(f"[{question_number} cached]")
            return cache[cache_key]

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=QUERY_PROMPT_TEMPLATE.format(question=question_text),
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


def generate_answers(question_df, df_type):
    cache_file = f"gpt_mmlu_pro_baseline_cache_{df_type}_answers.json"

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
        cat = example['category']
        ans = answer_question(q_text, idx, cat, cache, cache_file)
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

class QwenInference:
    def __init__(self, model_name="Qwen/Qwen2.5-7B"):
        """Initialize the model and tokenizer once"""
        print(f"Loading model {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        gc.collect()
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  
            trust_remote_code=True
        )

        print("Model loaded successfully!")
        
    def extract_answer_tags(self, response):
        """Extract content between <answer> and </answer> tags"""
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return response.strip()

    def answer_question(self, question_text, question_number, cache, cache_file, prompt_template):
        """Generate answer using the pre-loaded model"""
        
        # Check cache first
        cache_key = f"qwen::{question_text}"
        if cache_key in cache:
            print(f"[{question_number} cached]")
            return cache[cache_key]
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": prompt_template.format(question=question_text)
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
        return res

    def answer_batch(self, questions, cache, cache_file, prompt_template):
            """
            Generate answers for a batch of questions.
            Skips cached ones and saves results incrementally.
            """
            uncached = []
            prompts = []
            keys = []

            # Build batch prompts, skip cached
            for q in questions:
                cache_key = f"qwen::{q['question']}"
                keys.append(cache_key)
                if cache_key in cache:
                    prompts.append(None)
                else:
                    text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_template.format(question=q['question'])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(text)
                    uncached.append(True)
            
            # If everything is cached, just return from cache
            if not any(prompts):
                return [cache[k] for k in keys]

            # Tokenize only non-cached prompts
            inputs = self.tokenizer(
                [p for p in prompts if p is not None],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.6,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Merge cached + new results back into original order
            result_texts = []
            new_idx = 0
            for i, cache_key in enumerate(keys):
                if prompts[i] is None:
                    result_texts.append(cache[cache_key])
                else:
                    content = decoded[new_idx].strip()
                    new_idx += 1
                    cache[cache_key] = content
                    result_texts.append(content)
            
            # Save cache after batch
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)

            return result_texts

def process_questions_qwen(question_df, df_type, prompt_template, model_name, batch_size=8):
    cache_file = f"qwen_mmlu_pro_baseline_cache_{df_type}_{model_name}_answers.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    qwen_inference = QwenInference(f"Qwen/{model_name}")
    results = []

    total = len(question_df)
    for start in range(0, total, batch_size):
        batch = question_df[start:start+batch_size]
        answers = qwen_inference.answer_batch(batch, cache, cache_file, prompt_template)
        for q, ans in zip(batch, answers):
            results.append({
                'question': q['question'],
                'answer': ans
            })
        print(f"✓ Processed {min(start+batch_size, total)}/{total}")

    answer_df = pd.DataFrame(results, columns=["question", "answer"])

    del qwen_inference.model 
    del qwen_inference.tokenizer 
    del qwen_inference 
    gc.collect() 
    torch.cuda.empty_cache() 
    
    print("Processing complete.")
    return answer_df

def main():

    qual_test = pd.read_csv("../../datasets/mmlu/mmlu_pro_qualitative_test.csv")
    quant_test = pd.read_csv("../../datasets/mmlu/mmlu_pro_quantitative_test.csv")

    qual_valid = pd.read_csv("../../datasets/mmlu/mmlu_pro_qualitative_validation.csv")
    quant_valid = pd.read_csv("../../datasets/mmlu/mmlu_pro_quantitative_validation.csv")

    gpt_qual_test_answer_df = generate_answers(qual_test.to_dict(orient="records"), "qual_test")
    gpt_quant_test_answer_df = generate_answers(quant_test.to_dict(orient="records"), "quant_test")
    gpt_qual_valid_answer_df = generate_answers(qual_valid.to_dict(orient="records"), "qual_valid")
    gpt_quant_valid_answer_df = generate_answers(quant_valid.to_dict(orient="records"), "quant_valid")

    qwen_qual_test_answer_df = process_questions_qwen(qual_test.to_dict(orient="records"), "qual", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")
    qwen_quant_test_answer_df = process_questions_qwen(quant_test.to_dict(orient="records"), "quant", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")
    qwen_qual_valid_answer_df = process_questions_qwen(qual_valid.to_dict(orient="records"), "qual", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")
    qwen_quant_valid_answer_df = process_questions_qwen(quant_valid.to_dict(orient="records"), "quant", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")

    gpt_qual_test_answer_df.to_csv(f"gpt_mmlu_pro_baseline_qual_test_answers.csv", index=False)
    gpt_quant_test_answer_df.to_csv(f"gpt_mmlu_pro_baseline_quant_test_answers.csv", index=False)
    gpt_qual_valid_answer_df.to_csv(f"gpt_mmlu_pro_baseline_qual_valid_answers.csv", index=False)
    gpt_quant_valid_answer_df.to_csv(f"gpt_mmlu_pro_baseline_quant_valid_answers.csv", index=False)

    qwen_qual_test_answer_df.to_csv(f"qwen_mmlu_pro_baseline_qual_test_answers.csv", index=False)
    qwen_quant_test_answer_df.to_csv(f"qwen_mmlu_pro_baseline_quant_test_answers.csv", index=False)
    qwen_qual_valid_answer_df.to_csv(f"qwen_mmlu_pro_baseline_qual_valid_answers.csv", index=False)
    qwen_quant_valid_answer_df.to_csv(f"qwen_mmlu_pro_baseline_quant_valid_answers.csv", index=False)

if __name__ == "__main__":
    main()
    