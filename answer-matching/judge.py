### Libraries
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
import random

import answer_matching_prompts
from inference import HFInference
###
### Helpers 
MAX_WORKERS = 8

REASONING_OPENERS = [
    "Let's think step by step.",
    "We can reason this out.",
    "Consider the core claim.",
    "Reasoning through it:",
    "Breaking it down,",
    "Focusing on the key idea,",
    "Analyzing the options carefully,",
    "From first principles,",
    "In summary,",
    "Logically,"
]

PUNCT_PAD = [
    "",
    " ...",
    " —",
    " — —",
    " ... ..."
]

cache_lock = threading.Lock()

### Should we put this in a separate file?
### Surface manipulation
@dataclass
class SurfaceCfg:
    mode: str = "medium"        # light | medium | heavy
    add_opener: bool = True
    add_suffix_pad: bool = True
    seed: Optional[int] = None

def _rng(seed: Optional[int]):
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r

def surface_manipulate(answer: str, cfg: SurfaceCfg) -> str:
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)

    r = _rng(cfg.seed)

    opener = ""
    if cfg.add_opener:
        if cfg.mode == "light":
            opener = r.choice(REASONING_OPENERS[:4])
        elif cfg.mode == "medium":
            opener = r.choice(REASONING_OPENERS)
        else:  # heavy
            opener = r.choice(REASONING_OPENERS) + " " + r.choice(REASONING_OPENERS)

    suffix = ""
    if cfg.add_suffix_pad:
        if cfg.mode == "light":
            suffix = r.choice(PUNCT_PAD[:2])
        elif cfg.mode == "medium":
            suffix = r.choice(PUNCT_PAD)
        else:  # heavy
            suffix = r.choice(PUNCT_PAD + [" ... —", " — ..."])

    opener = (opener + " ").strip() + (" " if opener else "")
    return f"{opener}{answer}{suffix}"


###

def process_responses(
    responses_df,
    df_type,
    model,
    user_prompt,
    system_prompt=None,
    temperature=0.6,
    max_new_tokens=100,
    batch_size=4,
    # use_surface_attack: bool = False,
    # surface_mode: str = "medium",   # "light" | "medium" | "heavy"
    # surface_seed: Optional[int] = None,
):
    """
    Prompt for model judgement given question, answers and/or references.

    responses_df: list[dict] or DataFrame with columns/keys: question, reference, answer
    df_type:      'qual' | 'quant' | etc.
    model:        e.g. "Qwen3-4B"
    user_prompt:  prompt template string
    """

    # attack tag
    # attack_tag_cache = f"_surface-{surface_mode}" if use_surface_attack else ""
    # attack_tag_col   = f"surface:{surface_mode}" if use_surface_attack else "none"

    # namespace cache by attack so we don’t collide with baseline runs
    # cache_file = f"gpqa_diamond_cache_{df_type}_{model}_matches{attack_tag_cache}.json"
    cache_file = f"gpqa_diamond_cache_{df_type}_{model}_matches.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    # init inference backend
    if "qwen" in model.lower():
        mod_inference = HFInference(f"Qwen/{model}")
    else:
        mod_inference = HFInference(model)

    # support both DataFrame and list-of-dicts
    if isinstance(responses_df, pd.DataFrame):
        records = responses_df.to_dict(orient="records")
    else:
        records = responses_df

    results = []
    b_count = 0

    # config for surface manipulation (only used if flag is on)
    # cfg = SurfaceCfg(mode=surface_mode, seed=surface_seed)

    for start in range(0, len(records), batch_size):
        original_batch = records[start:start + batch_size]

        # make a transformed copy so we don't mutate the caller's data
        batch = []
        # for rec in original_batch:
            # rec2 = dict(rec)  
            # if use_surface_attack:
            #     # keep raw for audit; judge sees manipulated
            #     rec2["_answer_raw"] = rec2.get("answer", "")
            #     rec2["answer"] = surface_manipulate(rec2.get("answer", ""), cfg)
            # else:
            #     rec2["_answer_raw"] = rec2.get("answer", "")
            # batch.append(rec2)

        # run judge
        cache = mod_inference.generate_batch(
            batch, cache, cache_file, user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        # collect results
        for record in batch:
            cache_key = f"{mod_inference.model_name}::{record['question']}"
            score_val = cache[cache_key]['score']
            results.append({
                'judge': model,
                'df_type': df_type,
                # 'attack': attack_tag_col,         
                'question': record['question'],
                'reference': record['reference'],
                # 'response_raw': record['_answer_raw'], 
                'response': record['answer'],          
                'score': score_val
            })
            print(score_val)

        b_count += 1
        print(f"Proc {b_count} batch")

    # answer_df = pd.DataFrame(
    #     results,
        # columns=[
        #     "judge", "df_type", "attack",
        #     "question", "reference", "response_raw", "response", "score"
        # ]
    # )
    answer_df = pd.DataFrame(results, columns=["judge", "question", "reference", "response", "score"])


    # cleanup VRAM 
    del mod_inference.model
    del mod_inference.tokenizer
    del mod_inference
    gc.collect()
    torch.cuda.empty_cache()
    print("Processing complete.")
    return answer_df

def get_resp_df(q_df, a_df): 
    #original question df, generated responses df
    resp = q_df[['question', 'reference', 'question_mcq']].merge(a_df, on='question')
    resp = resp[['question', 'reference', 'answer']]
    return resp


def regenerate(path, num_tries, user_prompt_template, model_name, max_new_tokens=2048, temperature=0.01):
    """
    Clean up any corrupted scoring - 
    ie, cases where the matcher was not able to finish reasoning to get to scoring

    record: dict: {question, reference, answer}
    num_tries : max number of tries

    returns the corrected merged df
    """ 
    df = pd.read_csv(path)
    score_str = df["score"].astype(str)
    invalid = df[score_str.str.len() > 2]
    invalid =  invalid.rename(columns={"response": "answer"})
    records = invalid.to_dict(orient="records")

    if "qwen" in model_name.lower():
        mod_inference = HFInference(f"Qwen/{model_name}")

    corrected = []
    for i, record in enumerate(records):
        record = mod_inference.regenerate_resp(record, num_tries, user_prompt_template,
                                                max_new_tokens, temperature)
        corrected.append(record)

    corr_df = pd.DataFrame(corrected)

    base_path = path.split("/")[-1].split(".")[0] 
    valid_path = base_path + '_valid' + ".csv"
    # corr_df.to_csv(valid_path)
    print("Corrected!")
    old_df = df
    
    key = 'question'
    merged = old_df.set_index(key)
    merged.update(corr_df.set_index(key))
    
    merged = merged.reset_index()
    merged["flag_regen"] = merged[key].isin(corr_df[key])
    merged.to_csv(base_path + '_merged.csv', index=False)
    print("Merged!")

def main():
    user_prompt_template = answer_matching_prompts.get_judge_prompt_with_gt_baseline() #default: with COT

    #baseline datasets, baseline responses, baseline prompt judgement
    qual_df = pd.read_csv("../datasets/gpqa/gpqa_diamond_qualitative.csv")
    quant_df = pd.read_csv("../datasets/gpqa/gpqa_diamond_quantitative.csv")

    qwen_responses = pd.read_csv("../answer-generation/gpqa-baseline/gpqa_diamond_qual_qwen_answers.csv")
    gpt_responses = pd.read_csv("../answer-generation/gpqa-baseline/gpqa_diamond_qual_gpt_answers.csv")
    qual_responses_qwen = get_resp_df(qual_df, qwen_responses)
    qual_responses_gpt = get_resp_df(qual_df, gpt_responses)

    qwen_responses = pd.read_csv("../answer-generation/gpqa-baseline/gpqa_diamond_quant_qwen_answers.csv")
    gpt_responses = pd.read_csv("../answer-generation/gpqa-baseline/gpqa_diamond_quant_gpt_answers.csv")
    quant_responses_qwen = get_resp_df(quant_df, qwen_responses)
    quant_responses_gpt = get_resp_df(quant_df, gpt_responses)


    qwen_qual_am_df = process_responses(qual_responses_qwen.to_dict(orient="records"), "qual", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    qwen_quant_am_df = process_responses(quant_responses_qwen.to_dict(orient="records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    qwen_qual_am_df.to_csv("scores/gpqa/baseline/gpqa_diamond_qual_qwen_matches_baseline.csv", index=False)
    qwen_quant_am_df.to_csv("scores/gpqa/baseline/gpqa_diamond_quant_qwen_matches_baseline.csv", index=False)

    gpt_qual_am_df = process_responses(qual_responses_gpt.to_dict(orient="records"), "qual", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    gpt_quant_am_df = process_responses(quant_responses_gpt.to_dict(orient="records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048)
    gpt_qual_am_df.to_csv("scores/gpqa/baseline/gpqa_diamond_qual_gpt_matches_baseline.csv", index=False)
    gpt_quant_am_df.to_csv("scores/gpqa/baseline/gpqa_diamond_quant_gpt_matches_baseline.csv", index=False)

    #fix corrupted scores
    paths = ["scores/gpqa/baseline/gpqa_diamond_qual_gpt_matches_baseline.csv", 
     "scores/gpqa/baseline/gpqa_diamond_quant_gpt_matches_baseline.csv",
    "scores/gpqa/baseline/gpqa_diamond_qual_qwen_matches_baseline.csv",
    "scores/gpqa/baseline/gpqa_diamond_quant_qwen_matches_baseline.csv"]
    for p in paths:
        regenerate(p, 3, user_prompt_template, "Qwen3-4B")
        

    # # --- surface --- # DID NOT RUN YET
    # qwen_qual_surface_df  = process_responses(qual_responses_qwen.to_dict("records"),  "qual",  "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048, use_surface_attack=True, surface_mode="medium", surface_seed=7)
    # qwen_quant_surface_df = process_responses(quant_responses_qwen.to_dict("records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048, use_surface_attack=True, surface_mode="medium", surface_seed=7)
    # qwen_qual_surface_df.to_csv("gpqa_scores/gpqa_diamond_qual_qwen_matches_surface_medium.csv", index=False)
    # qwen_quant_surface_df.to_csv("gpqa_scores/gpqa_diamond_quant_qwen_matches_surface_medium.csv", index=False)

    # gpt_qual_surface_df  = process_responses(qual_responses_gpt.to_dict("records"),  "qual",  "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048, use_surface_attack=True, surface_mode="medium", surface_seed=7)
    # gpt_quant_surface_df = process_responses(quant_responses_gpt.to_dict("records"), "quant", "Qwen3-4B", user_prompt_template, temperature=0.01, max_new_tokens=2048, use_surface_attack=True, surface_mode="medium", surface_seed=7)
    # gpt_qual_surface_df.to_csv("gpqa_scores/gpqa_diamond_qual_gpt_matches_surface_medium.csv", index=False)
    # gpt_quant_surface_df.to_csv("gpqa_scores/gpqa_diamond_quant_gpt_matches_surface_medium.csv", index=False)

if __name__ == "__main__":
    main()
