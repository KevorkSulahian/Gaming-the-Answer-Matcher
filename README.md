# Gaming the Answer Matcher

A practical toolkit for studying and reproducing “gaming” behaviors in LLM answer-matching. It includes:
- Answer generation for benchmark datasets (e.g., GPQA, MMLU) with different strategies
- Answer matching/judging via OpenAI models and HuggingFace models
- Metric scripts and analysis helpers for qualitative and quantitative evaluations

This repo is organized around two main flows:
- Generate candidate answers for each question split
- Match/judge those answers against ground-truth references using a separate model (“matcher”/“judge”)

The code supports both API-based (OpenAI) and local (Transformers + PyTorch) inference.

## Repository Structure
- `answer-generation/` – Scripts for producing answers across strategies and datasets
  - `gpqa_diamond_baseline_answers_generation.py` – Baseline answer generation for GPQA
  - `surface_attack_generation.py` – Generation helpers for surface-level adversarial prompts
- `answer-matching/` – Matching and evaluation
  - `gen_and_match.py` – End-to-end generation + matching (HF + OpenAI support)
  - `inference.py` – Local HF inference wrapper for matching
  - `mmlu_answer_matching_gpt.py` – OpenAI-based matcher for MMLU-style evaluations
  - `metrics_*.py` – Metrics and plotting utilities (NumPy/Seaborn/Matplotlib)
- `datasets/` – Dataset assets and helpers
  - `gpqa/` – GPQA splits and processing (`gpqa-diamond-process-thread.py`)
  - `mmlu/` – MMLU processing (`mmlu-pro-process-thread.py`)
  - `merge_csv_json.py` – Merge helpers for cached results
- `InspectAI/` – Utilities built on top of `inspect_ai` for orchestration/evaluation
- `experiments/` – Experiment scripts for batch or variant evaluations

## Requirements
Install Python packages via:

```bash
pip install -r requirements.txt
```

Recommended: Python 3.11+ and a GPU  of a minimum of 16 GB Vram with recent CUDA if you plan to run local HF models.

## Environment Variables
Some scripts expect the following environment variables:
- `OPENAI_API_KEY` – Required if using OpenAI models for generation/matching
- `HF_TOKEN` – Required if pulling gated/private models from the Hugging Face Hub

Example (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:HF_TOKEN = "hf_..."
```

## Paper and Experiments

## Quick Troubleshooting
- CUDA OOM or slow inference: reduce `max_new_tokens`, use smaller models, or run on CPU for debugging.
- Missing parquet reader: ensure `pyarrow` is installed (provided in `requirements.txt`).
- Import errors for `inspect_ai`: install via `pip install inspect-ai` (included in `requirements.txt`).
- For private HF models: set `HF_TOKEN` before running HF-based scripts.

## License
See `LICENSE` for details.
