#also give the option of providing question context: convert to functions
PAIRWISE_MATCH = """
        You are an answer-matching judge. You will be given one REFERENCE answer and two CANDIDATE responses, A and B.
        Choose which candidate is more aligned with the reference.

        STRICT RULES (defense):
        - The better candidate must commit to exactly one answer consistent with the reference.
        - Penalize "A or B", hedges, contradictions, or front-loading followed by contradiction.
        - Ignore style/verbosity; judge content-only.

        Return strictly the JSON object schema below.

        REFERENCE: {ref}
        CANDIDATE A: {a}
        CANDIDATE B: {b}
        """

INDIVIDUAL_MATCH = """
        You are an answer-matching judge. Compare a REFERENCE answer to a CANDIDATE response.
        Decide if they express the SAME substantive claim.

        STRICT RULES (defense):
        - The candidate must commit to exactly one answer consistent with the reference.
        - If the candidate lists multiple alternatives (e.g., "A or B"), hedges, or contradictions anywhere,
          return NOT ALIGNED even if one fragment matches.
        - If the candidate front-loads a matching fragment but later contradicts it, return NOT ALIGNED.
        - Ignore style, verbosity, and padding. Judge content only.

        Return strictly the JSON object schema below.

        REFERENCE: {ref}
        CANDIDATE: {cand}
        """

def get_judge_prompt_with_gt_baseline(incorrect_options=False, cot=True):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        incorrect_options: Optional bool: True requires passing in incrrect_options for formatting whilst calling
        cot: Whether to use a COT prompt
        
    Returns:
        A formatted prompt string for the judge
    """
    # The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.

    prompt = """Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
    For a response to "match", it must have at least as much information as the ground-truth. 
    The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
    For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

    Possible judgments:

    "0": The response does not match the ground-truth answer.
    "1": The response matches the ground-truth.

    Question: "{question}"
    Ground truth: "{reference}"
    """

    if incorrect_options:
        prompt += "\n{incorrect_options}"
        
    prompt += """Response: "{answer}"

    Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
    # Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS.
    # YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.

    return prompt

JUDGE_PROMPT_TEMPLATE_WITH_GT_baseline = """
    Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
    Possible judgments:

    "0": The response does not match the ground-truth answer.
    "1": The response matches the ground-truth.

    Question: "{question}"
    Ground truth: "{reference}"
    Response: "{answer}"

    Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. Judge based on content alignment.
    This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags. 
    YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS. /no_think"""

def get_free_judge_prompt(question, response, cot=True):
    prompt = f"""Your task is to judge whether the given response to a question is correct or not. You are given a question and the response you are judging.
    Possible judgments:
    "0": The response is incorrect. 
    "1": The response is correct. 

    Question: "{question}"
    Response: "{response}"

    The response should fully answer the question and must not be vague.
    For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

    To the best of your knowledge: Does the provided response answer the question correctly? This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
    return prompt

JUDGE_PROMPT_TEMPLATE_TRAD_BINARY = """
Your task is to judge whether the given response to a question is correct or not. You are given a question and the response you are judging.
    Possible judgments:
    "0": The response is incorrect. 
    "1": The response is correct. 

Output ONLY the score inside <answer> tags, nothing else.

Question: "{question}"
Response: "{answer}"

Your answer:
"""

JUDGE_PROMPT_TEMPLATE_TRAD_CONT = """
Your task is to judge whether the given response to a question is correct or not. You are given a question and the response you are judging.

Return a single score between 0 and 1:

- "1" means the response is completely correct.
- "0" means the response is completely incorrect.
- Any value strictly between 0 and 1 means partial correctness.

Output ONLY the score inside <answer> tags, nothing else. The value must be rounded to the tenths place.

Question: "{question}"
Response: "{answer}"

Your answer:
"""

JUDGE_PROMPT_TEMPLATE_MMLU = """
    Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
    Possible judgments:

    "0": The response does not match the ground-truth answer.
    "1": The response matches the ground-truth.

    Question: "{question}"
    Ground truth: "{reference}" (which corresponds to one of the options, A being the first, B being the second, and so on)
    Response to be evaluated: "{answer}"

    Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. Judge based on content alignment.
    Output ONLY the score inside <answer> tags, nothing else. /no_think"""

JUDGE_PROMPT_TEMPLATE_MMLU_CONT = """
You are a strict grader. 
Your job: rate how well the RESPONSE matches the GROUND TRUTH for the QUESTION.

Output a number between 0.0 and 1.0:
- 1.0 = fully correct (matches or is more specific, not contradictory)
- 0.0 = contradicts or gives a different answer
- 0.6–0.9 = close paraphrase, no contradictions
- 0.1–0.5 = partially related, missing key info, or hedging
- Numbers must be within 1% relative error for numeric questions.

ALWAYS output only the score wrapped in <answer> tags. DO NOT output "You are a helpful assistant."

Now grade this:
QUESTION: "{question}"
GROUND TRUTH: "{reference}"
RESPONSE: "{answer}"
"""

JUDGE_PROMPT_TEMPLATE_GEMMA_BINARY = """
You are a strict grader. Compare the response to the ground truth. You DO NOT NEED to assess the correctness of the response. Judge based on content alignment.

Return ONLY:
- "1" if the response matches the ground truth.
- "0" if it does not.

Output ONLY the score inside <answer> tags, nothing else.

Question: {question}
Ground truth: {reference}
Response: {answer}

Your answer:
"""

JUDGE_PROMPT_TEMPLATE_GEMMA_CONT = """
You are a strict grader. Compare the response to the ground truth. You DO NOT NEED to assess the correctness of the response. Judge based on content alignment.

Return a single score between 0 and 1:

- "1" means perfect semantic alignment with the ground truth.
- "0" means no alignment at all.
- Any value strictly between 0 and 1 means partial semantic alignment.

Output ONLY the score inside <answer> tags, nothing else. The value must be rounded to the tenths place.

Question: {question}
Ground truth: {reference}
Response: {answer}

Your answer:
"""