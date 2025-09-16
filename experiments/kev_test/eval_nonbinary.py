#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-binary (0..1) answer-matching judge for ONE target (folder or CSV).
- Arg accepts either: a folder under answer-generation/gpqa/* OR a direct CSV path
- If folder: auto-pick one answers CSV (prefers QUAL/QUANT based on folder name)
- Auto-detects split (qual/quant) from CSV/folder name, can override via --split
- Continuous JSON: {"score": <float 0..1>, "reason": "<=20 words>"}
- tqdm progress + tokenizer pad hot-fix for decoder-only LMs
"""

# --- repo bootstrap: locate HFInference and REPO root ---
import pathlib
from importlib.machinery import SourceFileLoader

_here = pathlib.Path(__file__).resolve()
p = _here.parent
REPO = None
while p != p.parent:
    cand = p / "answer-matching" / "inference.py"
    if cand.exists():
        _inference = SourceFileLoader("am_inference", str(cand)).load_module()
        HFInference = _inference.HFInference
        REPO = p
        break
    p = p.parent
if REPO is None:
    raise RuntimeError("Couldn't locate answer-matching/inference.py")
# --- end bootstrap ---

# ===================== CONFIG =====================
from pathlib import Path
import os, re, json, gc, argparse
import pandas as pd
import torch
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm

DATASET_DIR = REPO / "datasets" / "gpqa"
SCORES_ROOT = REPO / "answer-matching" / "scores" / "gpqa_cont"

# judge model id (used by HFInference)
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
QUESTION_CONTEXT = None
TEMPERATURE = 0.01
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8   # safe for 4090 (can try 16)
# ==================================================

def _safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

# ------------------------- Continuous judge prompt -------------------------
_JSON_HEADER = """Return ONLY a single JSON object with this schema:
{{"score": <float 0..1>, "reason": "<=20 words>"}}

Rules for scores:
- 1.0 = semantically identical to ground truth and non-contradictory.
- 0.0 = contradicts / commits to a different answer / no commitment.
- Use (0.1..0.9) for partial paraphrase, missing detail, mild hedging.
- Penalize hedges, “A or B”, contradictions, final-line flip-flops.
- Ignore style/verbosity; judge content only.
- Think silently; output JSON ONLY.
"""

def _ctx(question: Optional[str], context: Optional[str]) -> str:
    lines = []
    if question: lines.append(f'QUESTION: "{question}"')
    if context:  lines.append(f'CONTEXT: "{context}"')
    return ("\n" + "\n".join(lines) + "\n") if lines else "\n"

CONTEXT_BLOCK = _ctx(None, QUESTION_CONTEXT)

USER_PROMPT_TEMPLATE = (
    "Your task is to judge whether the RESPONSE matches the GROUND TRUTH for the QUESTION.\n"
    "Score alignment on a continuous 0..1 scale.\n\n"
    "Guidance:\n"
    "- 1.0: fully matches (possibly more specific, non-contradictory).\n"
    "- 0.0: contradicts, commits to a different option, or no commitment.\n"
    "- (0.6..0.9): close paraphrase with minor omissions, no contradictions.\n"
    "- (0.1..0.5): partially related, missing key info, or hedging.\n"
    "- Numeric: require relative error < 1% for near-1.0; ranges ≠ single-point GT.\n\n"
    'QUESTION: "{question}"\n'
    'GROUND TRUTH: "{reference}"\n'
    'RESPONSE: "{answer}"\n\n'
    "Return ONLY a single JSON object:\n"
    '{{"score": <float 0..1>, "reason": "<=20 words>"}}\n'
    f"{CONTEXT_BLOCK}"
)


# ------------------------- Robust JSON parser -------------------------
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
def parse_continuous_json(text: str) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
    try:
        m = _JSON_BLOCK.search(text or "")
        if not m:
            return None, None, {}
        obj = json.loads(m.group(0))
        if "score" in obj:
            s = max(0.0, min(1.0, float(obj["score"])))
            return s, str(obj.get("reason", "")), obj
        return None, None, obj
    except Exception:
        return None, None, {}

# ------------------------- Data helpers -------------------------
def get_resp_df(q_df: pd.DataFrame, a_df: pd.DataFrame) -> pd.DataFrame:
    q = q_df.rename(columns={c: c.lower() for c in q_df.columns})
    a = a_df.rename(columns={c: c.lower() for c in a_df.columns})

    if "reference" not in q.columns and "answer" in q.columns:
        q = q.rename(columns={"answer": "reference"})
    if "answer" not in a.columns:
        for alt in ("response", "model_answer", "generated"):
            if alt in a.columns:
                a = a.rename(columns={alt: "answer"})
                break
    if "question" not in q.columns or "reference" not in q.columns:
        raise ValueError("Dataset CSV must have columns: question, reference")
    if "question" not in a.columns or "answer" not in a.columns:
        raise ValueError("Answers CSV must have columns: question, answer")

    m = q[["question", "reference"]].merge(a[["question", "answer"]], on="question")
    return m[["question", "reference", "answer"]]

def pick_one_answers_csv(folder: Path, split: str = "qual") -> Optional[Path]:
    tag = "qual" if split == "qual" else "quant"
    patterns = [
        f"*{tag}*qwen*answers*.csv",
        f"*{tag}*gpt*answers*.csv",
        f"*{tag}*answers*.csv",
        f"*{tag}*qwen*surface_medium*.csv",
        f"*{tag}*qwen*surface_heavy*.csv",
        f"*{tag}*qwen*surface_light*.csv",
        f"*{tag}*gpt*surface_medium*.csv",
        f"*{tag}*gpt*surface_heavy*.csv",
        f"*{tag}*gpt*surface_light*.csv",
        f"*{tag}*surface_medium*.csv",
        f"*{tag}*surface_heavy*.csv",
        f"*{tag}*surface_light*.csv",
        f"*{tag}*.csv",
    ]
    for pat in patterns:
        cands = sorted(folder.glob(pat))
        if cands:
            return cands[0]
    return None

def infer_split_from_path(path: Path) -> str:
    """Infer 'quant' or 'qual' from path; default to qual."""
    s = f"{path.name.lower()} {path.parent.name.lower()}"
    if "quant" in s:
        return "quant"
    if "qual" in s:
        return "qual"
    return "qual"

# ------------------------- Core evaluation -------------------------
def process_responses(records: List[Dict[str, Any]], cache_prefix: str, bar_desc: str) -> pd.DataFrame:
    cache_file = f"{cache_prefix}__{_safe_filename(JUDGE_MODEL)}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    else:
        cache = {}

    mod = HFInference(JUDGE_MODEL)

    # padding fix for decoder-only models
    tok = getattr(mod, "tokenizer", None)
    mdl = getattr(mod, "model", None)
    if tok is not None:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "left"
        if mdl is not None and getattr(mdl.config, "pad_token_id", None) in (None, -1):
            mdl.config.pad_token_id = tok.pad_token_id

    results = []
    pbar = tqdm(total=len(records), desc=bar_desc, unit="resp")
    for i in range(0, len(records), BATCH_SIZE):
        batch = [dict(rec) for rec in records[i:i+BATCH_SIZE]]
        cache = mod.generate_batch(
            batch, cache, cache_file,
            user_prompt=USER_PROMPT_TEMPLATE,
            system_prompt=None,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS
        )
        for rec in batch:
            cache_key = f"{mod.model_name}::{rec['question']}"
            entry = cache.get(cache_key, {})
            raw_text = entry.get("text") or entry.get("completion") or entry.get("score") or ""
            score, reason, _ = parse_continuous_json(raw_text)
            results.append({
                "judge": JUDGE_MODEL,
                "question": rec["question"],
                "reference": rec["reference"],
                "response": rec["answer"],
                "score": float("nan") if score is None else score,
                "reason": reason or "",
            })
        pbar.update(len(batch))
    pbar.close()

    df = pd.DataFrame(results, columns=["judge","question","reference","response","score","reason"])

    try:
        del mod.model; del mod.tokenizer; del mod
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return df

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Continuous 0–1 answer-matching judge for one target.")
    ap.add_argument("answers", help="Path to answers CSV OR a folder containing answers CSVs (baseline/surface).")
    ap.add_argument("--split", choices=["qual", "quant"], default=None,
                    help="Force GPQA split (overrides auto-detect).")
    ap.add_argument("--dataset-csv", default=None,
                    help="Explicit dataset CSV path (overrides split detection).")
    args = ap.parse_args()

    target = Path(args.answers)
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")

    # Resolve answers CSV
    if target.is_dir():
        split_guess = args.split or infer_split_from_path(target)
        csv = pick_one_answers_csv(target, split=split_guess)
        if not csv:
            raise FileNotFoundError(f"No suitable {split_guess.upper()} CSV found in {target}")
        folder_name = target.name
    else:
        csv = target
        folder_name = target.parent.name

    # Resolve dataset CSV
    if args.dataset_csv:
        dataset_csv = Path(args.dataset_csv)
        if not dataset_csv.exists():
            raise FileNotFoundError(f"--dataset-csv not found: {dataset_csv}")
        split_used = "custom"
    else:
        split_used = args.split or infer_split_from_path(csv)
        if split_used == "qual":
            dataset_csv = DATASET_DIR / "gpqa_diamond_qualitative.csv"
            if not dataset_csv.exists():
                cands = sorted(DATASET_DIR.glob("*qual*.csv"))
                if not cands:
                    raise FileNotFoundError(f"No QUAL dataset CSV found in {DATASET_DIR}")
                dataset_csv = cands[0]
        else:
            dataset_csv = DATASET_DIR / "gpqa_diamond_quantitative.csv"
            if not dataset_csv.exists():
                cands = sorted(DATASET_DIR.glob("*quant*.csv"))
                if not cands:
                    raise FileNotFoundError(f"No QUANT dataset CSV found in {DATASET_DIR}")
                dataset_csv = cands[0]

    print(f"[split]   {split_used}")
    print(f"[dataset] {dataset_csv}")
    print(f"[answers] {csv}")

    q_df = pd.read_csv(str(dataset_csv))
    a_df = pd.read_csv(str(csv))
    records = get_resp_df(q_df, a_df).to_dict(orient="records")

    out_dir = SCORES_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{csv.stem}__continuous_scores.csv"

    if out_csv.exists():
        print(f"✓ already exists, skipping: {out_csv}")
        return

    cache_prefix = f"gpqa_cache__{_safe_filename(folder_name)}__{_safe_filename(JUDGE_MODEL)}__{_safe_filename(str(split_used))}"
    out_df = process_responses(records, cache_prefix=cache_prefix, bar_desc=folder_name)
    out_df.to_csv(str(out_csv), index=False)
    print(f"✓ wrote: {out_csv}")

if __name__ == "__main__":
    main()
