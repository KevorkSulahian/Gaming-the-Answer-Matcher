### Libraries
import os
import json
import pandas as pd
import random
from dataclasses import dataclass
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

###
### Surface manipulation
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
### Data generation functions

def generate_surface_attacks(baseline_responses_df, surface_mode="medium", seed=42):
    """
    Generate surface attack versions of baseline responses
    
    Args:
        baseline_responses_df: DataFrame with columns ['question', 'reference', 'answer']
        surface_mode: 'light', 'medium', or 'heavy'
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with surface-manipulated answers
    """
    cfg = SurfaceCfg(mode=surface_mode, seed=seed)
    
    # Create a copy to avoid modifying original
    surface_df = baseline_responses_df.copy()
    
    # Apply surface manipulation to answers
    surface_df['answer_raw'] = surface_df['answer'].copy()
    surface_df['answer'] = surface_df['answer'].apply(lambda x: surface_manipulate(x, cfg))
    surface_df['surface_mode'] = surface_mode
    
    return surface_df

def load_baseline_responses(dataset_name='mmlu', df_type='qual', model='qwen'):
    """
    Load baseline responses from answer-generation folder
    
    Args:
        dataset_name: 'gpqa' or 'mmlu'
        df_type: 'qual' or 'quant'
        model: 'qwen' or 'gpt'
    
    Returns:
        DataFrame with baseline responses ready for surface attacks
    """
    # Load original dataset
    if dataset_name == 'gpqa':
        dataset_path = f"../datasets/gpqa/gpqa_diamond_{df_type}itative.csv"
        responses_path = f"gpqa/gpqa-baseline/gpqa_diamond_{df_type}_{model}_answers.csv"
    elif dataset_name == 'mmlu':
        dataset_path = f"../datasets/mmlu/mmlu_pro_{df_type}itative_test.csv"  
        responses_path = f"mmlu/mmlu-baseline/mmlu_pro_{df_type}_{model}_answers.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loading dataset from: {dataset_path}")
    print(f"Loading responses from: {responses_path}")
    
    # Check if files exist
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not os.path.exists(responses_path):
        raise FileNotFoundError(f"Responses file not found: {responses_path}")
    
    # Load dataset with questions and references
    dataset_df = pd.read_csv(dataset_path)
    
    # Load baseline responses
    responses_df = pd.read_csv(responses_path)
    
    print(f"Dataset columns: {dataset_df.columns.tolist()}")
    print(f"Responses columns: {responses_df.columns.tolist()}")
    
    # Merge to get question, reference, answer format
    if dataset_name == 'gpqa':
        merged_df = dataset_df[['question', 'reference', 'question_mcq']].merge(responses_df, on='question')
        result_df = merged_df[['question', 'reference', 'answer']]
    elif dataset_name == 'mmlu':
        # For MMLU, the correct answer is in the 'answer' column of dataset, response is in responses_df
        if 'response' in responses_df.columns:
            merged_df = dataset_df[['question', 'answer']].merge(responses_df[['question', 'response']], on='question')
            result_df = merged_df.rename(columns={'answer': 'reference', 'response': 'answer'})
        elif 'answer' in responses_df.columns:
            merged_df = dataset_df[['question', 'answer']].merge(responses_df[['question', 'answer']], on='question', suffixes=('_ref', '_resp'))
            result_df = merged_df[['question', 'answer_ref', 'answer_resp']].rename(columns={'answer_ref': 'reference', 'answer_resp': 'answer'})
        else:
            raise ValueError(f"Could not find response column in {responses_path}")
        
        result_df = result_df[['question', 'reference', 'answer']]
    
    print(f"Final result columns: {result_df.columns.tolist()}")
    print(f"Result shape: {result_df.shape}")
    
    return result_df

def generate_all_surface_attacks():
    """
    Generate surface attack data for all combinations of datasets, question types, and models
    """
    
    print("Generating surface attack data...")
    
    # datasets = ['gpqa']  # COMMENTED OUT GPQA
    datasets = ['mmlu']  # ADDED MMLU
    df_types = ['qual', 'quant'] 
    models = ['qwen', 'gpt']
    surface_modes = ['light', 'medium', 'heavy']
    
    for dataset in datasets:
        # Create dataset-specific output directory
        if dataset == 'mmlu':
            output_dir = f"mmlu/mmlu_surface_gaming"
        else:  # gpqa
            output_dir = f"gpqa/gpqa_surface_gaming"
        
        os.makedirs(output_dir, exist_ok=True)
        
        for df_type in df_types:
            for model in models:
                print(f"\nProcessing {dataset} {df_type} {model}...")
                
                try:
                    # Load baseline responses
                    baseline_df = load_baseline_responses(dataset, df_type, model)
                    print(f"Loaded {len(baseline_df)} baseline responses")
                    
                    # Generate surface attacks for each mode
                    for surface_mode in surface_modes:
                        print(f"  Generating {surface_mode} surface attacks...")
                        
                        surface_df = generate_surface_attacks(
                            baseline_df, 
                            surface_mode=surface_mode, 
                            seed=42
                        )
                        
                        # Save surface attack data with appropriate naming
                        if dataset == 'gpqa':
                            filename = f"gpqa_diamond_{df_type}_{model}_surface_{surface_mode}.csv"
                        elif dataset == 'mmlu':
                            filename = f"mmlu_pro_{df_type}_{model}_surface_{surface_mode}.csv"
                        
                        output_path = f"{output_dir}/{filename}"
                        surface_df.to_csv(output_path, index=False)
                        print(f"    Saved to {output_path}")
                        
                except FileNotFoundError as e:
                    print(f"  Skipping {dataset} {df_type} {model}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error processing {dataset} {df_type} {model}: {e}")
                    continue
    
    print("\nSurface attack data generation complete!")

def main():
    """Main function to generate all surface attack data"""
    
    print("=== Surface Attack Data Generation ===")
    
    # List available files first
    print("\nChecking available baseline files...")
    
    for dataset in ['mmlu']:  # Only checking MMLU now
        baseline_dir = f"{dataset}/{dataset}-baseline"
        if os.path.exists(baseline_dir):
            print(f"\n{dataset.upper()} baseline files:")
            for file in sorted(os.listdir(baseline_dir)):
                if file.endswith('.csv'):
                    print(f"  {file}")
        else:
            print(f"\n{dataset.upper()} baseline directory not found: {baseline_dir}")
    
    # Generate surface attacks on correct answers
    generate_all_surface_attacks()
    
if __name__ == "__main__":
    main()