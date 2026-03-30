# ==========================================================
# METRICS: BLEU + ROUGE + BERTScore (STABLE VERSION)
# ==========================================================

import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ==========================
# LOAD DATA
# ==========================

FILE_PATH = "outputs/multimodal_v2/test_predictions_v2.json"

with open(FILE_PATH) as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} samples")

# ==========================
# BLEU SETUP
# ==========================

smooth = SmoothingFunction().method1

bleu_scores = []

# ==========================
# ROUGE SETUP
# ==========================

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = []

# ==========================
# PREP FOR BERTSCORE
# ==========================

refs = []
preds = []

# ==========================
# LOOP
# ==========================

for item in tqdm(data, desc="Processing"):

    gt = item["ground_truth"].strip()
    pred = item["prediction"].strip()

    # skip empty cases
    if len(pred) == 0 or len(gt) == 0:
        continue

    # -------- BLEU --------
    bleu = sentence_bleu(
        [gt.split()],
        pred.split(),
        smoothing_function=smooth
    )
    bleu_scores.append(bleu)

    # -------- ROUGE --------
    scores = scorer.score(gt, pred)
    rouge_scores.append(scores["rougeL"].fmeasure)

    # -------- BERT PREP --------
    refs.append(gt)
    preds.append(pred)

# ==========================
# FINAL SCORES
# ==========================

avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge = sum(rouge_scores) / len(rouge_scores)

print("\n================ RESULTS ================\n")
print(f"BLEU Score   : {avg_bleu:.4f}")
print(f"ROUGE-L Score: {avg_rouge:.4f}")

# ==========================
# BERTSCORE (FIXED)
# ==========================

print("\n🔄 Computing BERTScore...")

P, R, F1 = bert_score(
    preds,
    refs,
    model_type="bert-base-uncased",   # ✅ FIXED MODEL
    lang="en",
    verbose=True
)

print("\n================ BERTScore ================\n")
print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1 Score:  {F1.mean().item():.4f}")