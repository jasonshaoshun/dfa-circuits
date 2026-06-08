TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "gpt2-small": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "gemma2": "google/gemma-2-2b",
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "llama3": "meta-llama/Llama-3.1-8B",
    "llama3-1b": "meta-llama/Llama-3.2-1B",
    "llama3-3b": "meta-llama/Llama-3.2-3B",
    "llama3-8b": "meta-llama/Llama-3.1-8B",
}

"""
This script will print a table of the following form:
Method      | IOI (GPT) | IOI (QWen) | IOI (Gemma) | IOI (Llama) | MCQA (QWen) | MCQA (Gemma) | MCQA (Llama) | Arithmetic (Llama) | ARC-E (Gemma) | ARC-E (Llama) | ARC-C (Llama)
Random      |
Method 1    |
Method 2    |
...
"""

COL_MAPPING = {
    "ioi_gpt2": 0, "ioi_qwen2.5": 1, "ioi_gemma2": 2, "ioi_llama3": 3,
    "mcqa_qwen2.5": 4, "mcqa_gemma2": 5, "mcqa_llama3": 6,
    "arithmetic-addition_llama3": 7, "arithmetic-subtraction_llama3": 8,
    "arc-easy_gemma2": 9, "arc-easy_llama3": 10,
    "arc-challenge_llama3": 11,
    "ioi_interpbench": None,
    "ioi_llama3-1b": 12, "ioi_llama3-3b": 13, "ioi_llama3-8b": 14, "ioi_qwen2.5-0.5b": 15, "ioi_qwen2.5-1.5b": 16, "ioi_qwen2.5-3b": 17, "ioi_gemma2-2b": 18, "ioi_gemma2-9b": 19, "ioi_gemma2-27b": 20,
    "mcqa_llama3-1b": 21, "mcqa_llama3-3b": 22, "mcqa_llama3-8b": 23, "mcqa_qwen2.5-0.5b": 24, "mcqa_qwen2.5-1.5b": 25, "mcqa_qwen2.5-3b": 26, "mcqa_gemma2-2b": 27, "mcqa_gemma2-9b": 28, "mcqa_gemma2-27b": 29,
    "arithmetic-addition_llama3-1b": 30, "arithmetic-addition_llama3-3b": 31, "arithmetic-addition_llama3-8b": 32, "arithmetic-addition_qwen2.5-0.5b": 33, "arithmetic-addition_qwen2.5-1.5b": 34, "arithmetic-addition_qwen2.5-3b": 35, "arithmetic-addition_gemma2-2b": 36, "arithmetic-addition_gemma2-9b": 37, "arithmetic-addition_gemma2-27b": 38,
    "arithmetic-subtraction_llama3-1b": 39, "arithmetic-subtraction_llama3-3b": 40, "arithmetic-subtraction_llama3-8b": 41, "arithmetic-subtraction_qwen2.5-0.5b": 42, "arithmetic-subtraction_qwen2.5-1.5b": 43, "arithmetic-subtraction_qwen2.5-3b": 44, "arithmetic-subtraction_gemma2-2b": 45, "arithmetic-subtraction_gemma2-9b": 46, "arithmetic-subtraction_gemma2-27b": 47,
    "arc-easy_llama3-1b": 48, "arc-easy_llama3-3b": 49, "arc-easy_llama3-8b": 50, "arc-easy_qwen2.5-0.5b": 51, "arc-easy_qwen2.5-1.5b": 52, "arc-easy_qwen2.5-3b": 53, "arc-easy_gemma2-2b": 54, "arc-easy_gemma2-9b": 55, "arc-easy_gemma2-27b": 56,
    "arc-challenge_llama3-1b": 57, "arc-challenge_llama3-3b": 58, "arc-challenge_llama3-8b": 59, "arc-challenge_qwen2.5-0.5b": 60, "arc-challenge_qwen2.5-1.5b": 61, "arc-challenge_qwen2.5-3b": 62, "arc-challenge_gemma2-2b": 63, "arc-challenge_gemma2-9b": 64, "arc-challenge_gemma2-27b": 65,
    "ioi_gpt2-small": 66, "ioi_gpt2-medium": 67, "ioi_gpt2-large": 68,
    "mcqa_gpt2-small": 69, "mcqa_gpt2-medium": 70, "mcqa_gpt2-large": 71,
    "arithmetic-addition_gpt2-small": 72, "arithmetic-addition_gpt2-medium": 73, "arithmetic-addition_gpt2-large": 74,
    "arithmetic-subtraction_gpt2-small": 75, "arithmetic-subtraction_gpt2-medium": 76, "arithmetic-subtraction_gpt2-large": 77,
    "arc-easy_gpt2-small": 78, "arc-easy_gpt2-medium": 79, "arc-easy_gpt2-large": 80,
    "arc-challenge_gpt2-small": 81, "arc-challenge_gpt2-medium": 82, "arc-challenge_gpt2-large": 83,
}

# header = ["Method", "IOI (GPT)", "IOI (QWen)", "IOI (Gemma)", "IOI (Llama)", "MCQA (QWen)", "MCQA (Gemma)", "MCQA (Llama)",
#             "Arithmetic (Llama)", "ARC-E (Gemma)", "ARC-E (Llama)", "ARC-C (Llama)"]
# header = [
#     "Method",
#     "IOI (GPT)", "IOI (QWen)", "IOI (Gemma)", "IOI (Llama)",
#     "MCQA (QWen)", "MCQA (Gemma)", "MCQA (Llama)",
#     "Arithmetic Addition (Llama)", "Arithmetic Subtraction (Llama)",
#     "ARC-Easy (Gemma)", "ARC-Easy (Llama)",
#     "ARC-Challenge (Llama)",
#     "IOI (Llama3-1B)", "IOI (Llama3-3B)", "IOI (GPT2)", "IOI (GPT2-Medium)", "IOI (GPT2-Large)",
#     "MCQA (Llama3-1B)", "MCQA (Llama3-3B)",
#     "Arithmetic Addition (Llama3-1B)", "Arithmetic Addition (Llama3-3B)",
#     "Arithmetic Subtraction (Llama3-1B)", "Arithmetic Subtraction (Llama3-3B)",
#     "ARC-Easy (Llama3-1B)", "ARC-Easy (Llama3-3B)",
#     "ARC-Challenge (Llama3-1B)", "ARC-Challenge (Llama3-3B)"
# ]
