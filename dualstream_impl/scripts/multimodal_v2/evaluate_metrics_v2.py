# ==========================================================
# METRICS EVALUATION (BLEU + ROUGE)
# ==========================================================

import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ==========================
# LOAD PREDICTIONS
# ==========================

PRED_FILE = "outputs/multimodal_v2/test_predictions_v2.json"

with open(PRED_FILE) as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} samples")

# ==========================
# METRIC SETUP
# ==========================

smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

bleu_scores = []
rouge1_scores = []
rougeL_scores = []

# ==========================
# LOOP
# ==========================

for item in tqdm(data):

    gt = item["ground_truth"].strip()
    pred = item["prediction"].strip()

    # -------- BLEU --------
    bleu = sentence_bleu(
        [gt.split()],
        pred.split(),
        smoothing_function=smooth
    )
    bleu_scores.append(bleu)

    # -------- ROUGE --------
    scores = scorer.score(gt, pred)

    rouge1_scores.append(scores["rouge1"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

# ==========================
# FINAL SCORES
# ==========================

avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print("\n================ RESULTS ================\n")
print(f"BLEU Score   : {avg_bleu:.4f}")
print(f"ROUGE-1 Score: {avg_rouge1:.4f}")
print(f"ROUGE-L Score: {avg_rougeL:.4f}")