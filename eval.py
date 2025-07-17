import json
import re
import string
from collections import Counter
from typing import List, Tuple, Union


def normalize_text(text: str) -> str:
    """
    Normalize text by removing punctuation, converting to lowercase,
    and removing extra whitespace.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace after normalization.
    """
    normalized = normalize_text(text)
    return normalized.split()


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Calculate Exact Match score between prediction and ground truth.
    
    Args:
        prediction: The predicted answer
        ground_truth: The ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_normalized = normalize_text(prediction)
    gt_normalized = normalize_text(ground_truth)

    return 1.0 if pred_normalized == gt_normalized else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score between prediction and ground truth based on token overlap.
    
    Args:
        prediction: The predicted answer
        ground_truth: The ground truth answer
        
    Returns:
        F1 score (float between 0 and 1)
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    # Handle empty cases
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # Count token frequencies
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # Calculate intersection (common tokens)
    common_tokens = pred_counter & gt_counter
    num_common = sum(common_tokens.values())

    # Calculate precision and recall
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_batch(predictions: List[str],
                   ground_truths: List[str]) -> Tuple[float, float]:
    """
    Evaluate a batch of predictions against ground truths.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Tuple of (average_em, average_f1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Number of predictions must match number of ground truths")

    em_scores = []
    f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        em_scores.append(exact_match(pred, gt))
        f1_scores.append(f1_score(pred, gt))

    avg_em = sum(em_scores) / len(em_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    return avg_em, avg_f1


def evaluate_single(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        prediction: The predicted answer
        ground_truth: The ground truth answer
        
    Returns:
        Tuple of (em_score, f1_score)
    """
    em = exact_match(prediction, ground_truth)
    f1 = f1_score(prediction, ground_truth)

    return em, f1


k = 3
data_folder = '2WikiMultihopQA'  # '2WikiMultihopQA', 'HotpotQA', 'CloudComputing'
data_extenstion = '-full-list'  # '-full-list', '-dev-context'
llm_provider = 'gemini'  # 'openai', 'ollama', 'huggingface', 'gemini'
mode = 'rag+related_data'  # 'related_data', 'rag+related_data', 'rag&related_data', 'rag'
embedder = 'MiniLM-L6-150'

if 'rag' in mode:
    file_path = f'./Data/{data_folder}/Results/{embedder}/results{data_extenstion}-{mode}-{llm_provider}-k{k}.json'
else:
    file_path = f'./Data/{data_folder}/Results/{embedder}/results{data_extenstion}-{mode}-{llm_provider}.json'

index = []
predictions = []
ground_truths = []
correct_count = 0
unknown_count = 0
wrong_count = 0
with open(file_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

for item in results:
    predictions.append(item['result'])
    ground_truths.append(item['answer'])
    answer = item['answer'].lower().strip().replace('?', '').replace('.', '')
    output = item['result'].lower().strip().replace('?', '').replace('.', '')

    if output.lower() == "unknown":
        unknown_count += 1
    elif answer == output:
        correct_count += 1
    else:
        wrong_count += 1
        index.append(item['id'])

print(file_path)
avg_em, avg_f1 = evaluate_batch(predictions, ground_truths)
print(f"Correct answers: {correct_count}")
print(f"Wrong answers: {wrong_count}")
print(f"Unknown answers: {unknown_count}")
print(f"Average Exact Match: {avg_em:.3f}")
print(f"Average F1 Score: {avg_f1:.3f}")
print(f'To-check indexes: {index}')
