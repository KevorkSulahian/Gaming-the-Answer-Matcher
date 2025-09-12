#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-binary (0..1) answer-matching judge for ONE target (folder or CSV), chosen via CLI arg.
- Arg accepts either: a folder under answer-generation/gpqa/* OR a direct CSV path
- If folder: auto-pick one QUAL CSV (supports baseline and surface_* files)
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

DATASET_DIR  = REPO / "datasets" / "gpqa"
SCORES_ROOT  = REPO / "answer-matching" / "scores" / "gpqa"

# dataset: qualitative split (edit to use quant if needed)
DATASET_CSV = DATASET_DIR / "gpqa_diamond_qualitative.csv"
if not DATASET_CSV.exists():
    cands = sorted(DATASET_DIR.glob("*qual*.csv"))
    if not cands:
        raise FileNotFoundError(f"No QUAL dataset CSV found in {DATASET_DIR}")
    DATASET_CSV = cands[0]

# judge model id (used by HFInference)
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
QUESTION_CONTEXT = None
TEMPERATURE = 0.01
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 4
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

# ---- minimal picker (supports baseline + surface names) ----
def pick_one_answers_csv(folder: Path) -> Optional[Path]:
    """
    Minimal priority:
      1) *qual*qwen*answers*.csv
      2) *qual*gpt*answers*.csv
      3) *qual*answers*.csv
      4) *qual*qwen*surface_(medium|heavy|light).csv
      5) *qual*gpt*surface_(medium|heavy|light).csv
      6) *qual*surface_(medium|heavy|light).csv
      7) any *qual*.csv
    Reorder lines to flip preferences.
    """
    patterns = [
        "*qual*qwen*answers*.csv",
        "*qual*gpt*answers*.csv",
        "*qual*answers*.csv",

        "*qual*qwen*surface_medium*.csv",
        "*qual*qwen*surface_heavy*.csv",
        "*qual*qwen*surface_light*.csv",

        "*qual*gpt*surface_medium*.csv",
        "*qual*gpt*surface_heavy*.csv",
        "*qual*gpt*surface_light*.csv",

        "*qual*surface_medium*.csv",
        "*qual*surface_heavy*.csv",
        "*qual*surface_light*.csv",

        "*qual*.csv",
    ]
    for pat in patterns:
        cands = sorted(folder.glob(pat))
        if cands:
            return cands[0]
    return None
# -------------------------------------------------------------

# ------------------------- Core evaluation -------------------------
def process_responses(records: List[Dict[str, Any]], cache_prefix: str, bar_desc: str) -> pd.DataFrame:
    cache_file = f"{cache_prefix}__{_safe_filename(JUDGE_MODEL)}.json"

    # load cache if exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    else:
        cache = {}

    # init judge backend
    mod = HFInference(JUDGE_MODEL)

    # --- padding hot-fix for decoder-only models (e.g., LLaMA) ---
    tok = getattr(mod, "tokenizer", None)
    mdl = getattr(mod, "model", None)
    if tok is not None:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "left"
        if mdl is not None and getattr(mdl.config, "pad_token_id", None) in (None, -1):
            mdl.config.pad_token_id = tok.pad_token_id
    # -------------------------------------------------------------

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

    # cleanup VRAM
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
    args = ap.parse_args()

    target = Path(args.answers)
    if not target.exists():
        raise FileNotFoundError(f"{target} does not exist")

    # resolve CSV
    if target.is_dir():
        csv = pick_one_answers_csv(target)
        if not csv:
            raise FileNotFoundError(f"No suitable QUAL CSV found in {target}")
        folder_name = target.name
    else:
        csv = target
        folder_name = target.parent.name

    print(f"[dataset] {DATASET_CSV}")
    print(f"[answers] {csv}")

    # read data
    q_df = pd.read_csv(str(DATASET_CSV))
    a_df = pd.read_csv(str(csv))
    records = get_resp_df(q_df, a_df).to_dict(orient="records")

    # process
    cache_prefix = f"gpqa_cache__{_safe_filename(folder_name)}"
    out_df = process_responses(records, cache_prefix=cache_prefix, bar_desc=folder_name)

    # output
    out_dir = SCORES_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{csv.stem}__continuous_scores.csv"
    out_df.to_csv(str(out_csv), index=False)
    print(f"✓ wrote: {out_csv}")

if __name__ == "__main__":
    main()
