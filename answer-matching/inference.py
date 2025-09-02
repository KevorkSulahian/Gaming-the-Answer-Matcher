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



class HFInference:
    def __init__(self, model_name, device_map='auto'):
        print(f"Loading model {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device_map} if isinstance(device_map, str) and device_map.startswith("cuda") else device_map,
            trust_remote_code=True
        )
        print(f"Model {model_name} loaded on device: {self.model.device}")

        print("Model loaded successfully!")

    def extract_answer_tags(self, response):
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()

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
            try_i += 0

        record['score'] = res
        print(record['score'])
            
        return record


