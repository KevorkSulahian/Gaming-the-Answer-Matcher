import pandas as pd

def main():
    # mmlu_pro_qual = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test.csv")
    # mmlu_pro_quant = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_test.csv")

    mmlu_pro_qual_sample = pd.read_csv("datasets/mmlu/mmlu_pro_qualitative_test_sample_large.csv")
    mmlu_pro_quant_sample = pd.read_csv("datasets/mmlu/mmlu_pro_quantitative_test_sample_large.csv")

    mmlu_pro_qual_answers = pd.read_csv("answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_qual_test_answers.csv")
    mmlu_pro_quant_answers = pd.read_csv("answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_quant_test_answers.csv")

    # Define patterns that indicate dependence on answer choices
    patterns = [
        "which of the following",
        "all of the above",
        "none of the above",
        "both a and b",
        "both b and c",
        "all of these",
        "one of the following",
        "except",
        "is/are correct",
        "is true",
        "is not true",
        "is false"
    ]

    # Build regex from patterns
    regex = "|".join(patterns)

    # Filter out dependent questions
    mmlu_pro_qual_sample_filtered = mmlu_pro_qual_sample[
        ~mmlu_pro_qual_sample["question"].str.lower().str.contains(regex, na=False)
    ]
    mmlu_pro_quant_sample_filtered = mmlu_pro_quant_sample[
        ~mmlu_pro_quant_sample["question"].str.lower().str.contains(regex, na=False)
    ]

    sample_qual_questions = mmlu_pro_qual_sample_filtered["question"].tolist()
    sample_quant_questions = mmlu_pro_quant_sample_filtered["question"].tolist()

    answers_sample_qual = mmlu_pro_qual_answers[mmlu_pro_qual_answers['question'].isin(sample_qual_questions)]
    answers_sample_quant = mmlu_pro_quant_answers[mmlu_pro_quant_answers['question'].isin(sample_quant_questions)]

    # Save filtered answers
    # answers_sample_qual.to_csv(
    #     "answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_qual_test_answers_sample_large_filtered.csv",
    #     index=False
    # )
    # answers_sample_quant.to_csv(
    #     "answer-generation/mmlu/mmlu-verbose/gpt_mmlu_pro_verbose_quant_test_answers_sample_large_filtered.csv",
    #     index=False
    # )

    mmlu_pro_qual_sample_filtered.to_csv("mmlu_pro_qual_filtered.csv")
    mmlu_pro_quant_sample_filtered.to_csv("mmlu_pro_quant_filtered.csv")

if __name__ == "__main__":
    main()
