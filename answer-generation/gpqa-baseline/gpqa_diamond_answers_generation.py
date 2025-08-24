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


openai_client = OpenAI() #ensure you define your openai api key
hf_client = InferenceClient() #ensure hf_token w/ finegrained inference permission is defined
cache_lock = threading.Lock()

MAX_WORKERS = 8

QUERY_PROMPT_TEMPLATE = """
You will be asked a question. Please provide your answer as a free-text response that is 3-4 sentences long. Keep your answer concise and to the point. Do not include any additional information or context beyond what is necessary to answer the question.

Question: {question}
"""
# Put your final answer in <answer> </answer> tags.
# Your final answer should be concise and your response SHOULD STRICTLY BE ENCLOSED IN <answer> </answer> tags.


def answer_question_gpt(question_text, question_number, cache, cache_file, model="gpt-4.1-mini"):
    cache_key = f"{question_text}"
    with cache_lock:
        if cache_key in cache:
            # Just a small, single-line progress print
            print(f"[{question_number} cached]")
            return cache[cache_key]
    
    response = openai_client.responses.create(
        model=model,
        input=QUERY_PROMPT_TEMPLATE.format(question=question_text),
        max_output_tokens=300,
        temperature=0
    )


    text_response = response.output_text.strip()

    with cache_lock:
        cache[cache_key] = text_response
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        # print(f"[{question_number} new]")

    return text_response


def generate_answers(question_df, df_type, model):
    cache_file = f"gpqa_diamond_cache_{df_type}_{model}_answers.json"

    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Collect questions to process (skip ones already cached)
    to_process = [(i, ex) for i, ex in enumerate(question_df)
                  if f"{ex['question']}" not in cache]

    total = len(to_process)
    progress = [0]  # list so it can be mutated inside threads

    # For storing answers in a dict keyed by index to preserve order
    indexed_results = {}

    def process_item(item):
        idx, example = item
        q_text = example['question']
        if 'gpt' in model.lower():
            ans = answer_question_gpt(q_text, idx, cache, cache_file, model)
        with cache_lock:
            progress[0] += 1
            # print(f"✓ Answered question {progress[0]}/{total}")
            cache[f"{q_text}"] = ans
            indexed_results[idx] = ( q_text, ans)
        return ans

    # Process in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_item, item): item for item in to_process}
        for future in tqdm(as_completed(futures), total=len(to_process), desc="Processing"):
            try:
                future.result()
            except Exception as e:
                idx, _ = futures[future]
                print(f"Error processing question {idx}: {e}")

    # Add already cached items to results
    for i, ex in enumerate(question_df):
        key = f"{ex['question']}"
        if key in cache:
            indexed_results[i] = (ex['question'], cache[key])

    # Save updated cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Sort by original question order
    sorted_results = [indexed_results[i] for i in sorted(indexed_results)]
    answer_df = pd.DataFrame(sorted_results, columns=["question", "answer"])
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
        
        # # Check which GPU has which layers
        # for name, module in self.model.named_modules():
        #     if hasattr(module, 'weight') and module.weight is not None:
        #             print(f"{name}: {module.weight.device}")


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

def process_questions_qwen(question_df, df_type, prompt_template, model):
    cache_file = f"gpqa_diamond_cache_{df_type}_{model}_answers.json"
    # Load cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    qwen_inference = QwenInference(f"Qwen/{model}")
    
    results = []
    
    # Process all questions with the same model instance
    for i, question in enumerate(question_df):
        answer = qwen_inference.answer_question(
            question['question'], 
            i + 1, 
            cache, 
            cache_file, 
            prompt_template
        )
        results.append({
            'question': question['question'],
            'answer': answer
        })
        
    answer_df = pd.DataFrame(results, columns=["question", "answer"])
    print("Processing complete.")
    return answer_df



def main():
    qual = pd.read_csv("../../datasets/gpqa/gpqa_diamond_qualitative.csv")
    quant = pd.read_csv("../../datasets/gpqa/gpqa_diamond_quantitative.csv")

    gpt_qual_answer_df = generate_answers(qual.to_dict(orient="records"), "qual", "gpt-4.1-mini")
    gpt_quant_answer_df = generate_answers(quant.to_dict(orient="records"), "quant", "gpt-4.1-mini")

    gpt_qual_answer_df.to_csv(f"gpqa_diamond_qual_gpt_answers.csv", index=False)
    gpt_quant_answer_df.to_csv(f"gpqa_diamond_quant_gpt_answers.csv", index=False)


    qwen_qual_answer_df = process_questions_qwen(qual.to_dict(orient="records"), "qual", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")
    qwen_quant_answer_df = process_questions_qwen(quant.to_dict(orient="records"), "quant", QUERY_PROMPT_TEMPLATE, "Qwen2.5-7B-Instruct")

    qwen_qual_answer_df.to_csv(f"gpqa_diamond_qual_qwen_answers.csv", index=False)
    qwen_quant_answer_df.to_csv(f"gpqa_diamond_quant_qwen_answers.csv", index=False)

if __name__ == "__main__":
    main() 