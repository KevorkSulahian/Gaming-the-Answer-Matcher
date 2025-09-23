import os

HF_CACHE = "/workspace/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE  # optional, for datasets
os.environ["HF_TOKEN"] = "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"
os.makedirs(HF_CACHE, exist_ok=True)

import json
import re
import gc
import torch
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient

# HF cache setup

class HFInference:
    def __init__(self, model_name, device_map='auto'):
        print(f"Loading model {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=os.environ["HF_TOKEN"]
        )
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device_map} if isinstance(device_map, str) and device_map.startswith("cuda") else device_map,
            trust_remote_code=True,
            token=os.environ["HF_TOKEN"]
        )
        print(f"Model {model_name} loaded on device: {self.model.device}")
        print("Model loaded successfully!")

    def extract_numeric_score(self, response):
        """
        Extract numeric value from <answer>...</answer> in response.
        Returns a float in [0,1]. Defaults to 0.0 if parsing fails.
        """
        match = re.search(r"<answer>\s*([0-9.]+)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
        if match:
            raw_val = match.group(1).strip()
            try:
                val = float(raw_val)
                return max(0.0, min(1.0, val))  # clamp to [0,1]
            except ValueError:
                return 0.0
        return 0.0

    def extract_answer_tags(self, response):
        """
        Extracts content inside <answer> tags.
        Returns the inner string or full response if tag not found.
        """
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()

    def generate_batch(
        self,
        batch,
        cache,
        cache_file,
        user_prompt,
        system_prompt=None,
        temperature=0.6,
        max_new_tokens=128,
        force_regen=False  # added flag to recompute even if cached
    ):
        new_cache = {}

        for record in batch:
            cache_key = f"{self.model_name}::{record['question']}"
            if cache_key in cache and not force_regen:
                continue  # skip already processed

            prompt = user_prompt.format(
                question=record["question"],
                reference=record["reference"],
                answer=record["answer"]
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = self.extract_numeric_score(text)

            new_cache[cache_key] = {
                "prompt": prompt,
                "response": text,
                "score": score,
            }

        cache.update(new_cache)
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

        return cache

    def regenerate_resp(self, record, num_tries, user_prompt_template, max_new_tokens=2048,
                        temperature=0.01):
        """
        Regenerate response until a non-empty <answer> is found or num_tries exhausted.
        Returns record with numeric score.
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
        res_score = 0.0
        while try_i < num_tries:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            extracted = self.extract_answer_tags(content)
            res_score = self.extract_numeric_score(extracted)

            if res_score != 0.0:  # stop if valid score found
                break

            try_i += 1

        record['score'] = res_score
        print(record['score'])
        return record
