# ==========================================================
# DOMAIN-SPECIFIC EVALUATION (SEMANTIC + HUMAN-LIKE)
# ==========================================================

import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# CONFIG
# ==========================

FILE_PATH = "outputs/multimodal_v2/test_predictions_v2.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("✅ Device:", device)

# ==========================
# LOAD DATA
# ==========================

with open(FILE_PATH) as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} samples")

# ==========================
# LOAD EVALUATOR MODEL
# ==========================

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16
).to(device)

llm.eval()

print("✅ Evaluator loaded")

# ==========================
# SEMANTIC PROMPT (CORE FIX)
# ==========================

def build_prompt(gt, pred, task):

    if "caption" in task:
        return f"""
You are a human expert evaluating disaster damage reports.

IMPORTANT RULES:
- Ignore wording differences
- Ignore sentence order
- Ignore formatting differences
- Focus ONLY on semantic meaning and correctness

GROUND TRUTH:
{gt}

PREDICTION:
{pred}

Evaluate based on MEANING:

DAP (Damage Assessment Precision):
→ Are the correct damages identified?

DDR (Damage Detail Recall):
→ Are all important elements covered (building, road, vegetation, etc.)?

FC (Factual Correctness):
→ Is the prediction logically correct without hallucination?

Score each from 1 (poor) to 5 (excellent)

Return ONLY:

DAP: <number>
DDR: <number>
FC: <number>
"""

    else:
        return f"""
You are a human expert evaluating disaster recovery plans.

IMPORTANT RULES:
- Ignore wording differences
- Ignore order of sentences
- Focus ONLY on semantic usefulness and reasoning

GROUND TRUTH:
{gt}

PREDICTION:
{pred}

Evaluate based on MEANING:

RN (Recovery Necessity):
→ Are the suggested actions relevant and meaningful?

SC (Strategy Completeness):
→ Does it cover both immediate and long-term recovery?

APP (Action Priority Precision):
→ Are actions logically prioritized?

Score each from 1 (poor) to 5 (excellent)

Return ONLY:

RN: <number>
SC: <number>
APP: <number>
"""

# ==========================
# ROBUST PARSER
# ==========================

def extract_number(text, key):
    try:
        match = re.search(rf"{key}\s*[:\-]?\s*(\d+(\.\d+)?)", text)
        if match:
            return float(match.group(1))
        return None
    except:
        return None


def parse_scores(text, task):

    if "caption" in task:
        dap = extract_number(text, "DAP")
        ddr = extract_number(text, "DDR")
        fc = extract_number(text, "FC")

        if None in (dap, ddr, fc):
            return None
        return dap, ddr, fc

    else:
        rn = extract_number(text, "RN")
        sc = extract_number(text, "SC")
        app = extract_number(text, "APP")

        if None in (rn, sc, app):
            return None
        return rn, sc, app

# ==========================
# EVALUATION LOOP
# ==========================

cap_scores = []
adv_scores = []

debug_count = 0
skipped = 0

for item in tqdm(data):

    gt = item["ground_truth"]
    pred = item["prediction"]
    task = item.get("task", "caption")

    prompt = build_prompt(gt, pred, task)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    scores = parse_scores(response, task)

    # 🔍 DEBUG OUTPUT
    if debug_count < 5:
        print("\n================ DEBUG =================")
        print(response)
        debug_count += 1

    if scores is None:
        skipped += 1
        continue

    if "caption" in task:
        cap_scores.append(scores)
    else:
        adv_scores.append(scores)

# ==========================
# FINAL RESULTS
# ==========================

print("\n================ DOMAIN RESULTS ================\n")

print(f"⚠️ Skipped samples (parse fail): {skipped}")

# -------- CAPTION --------
if cap_scores:
    dap = sum(x[0] for x in cap_scores) / len(cap_scores)
    ddr = sum(x[1] for x in cap_scores) / len(cap_scores)
    fc  = sum(x[2] for x in cap_scores) / len(cap_scores)

    print("\n📌 CAPTION TASK (Semantic Evaluation):")
    print(f"DAP (Damage Precision): {dap:.3f}")
    print(f"DDR (Detail Recall)   : {ddr:.3f}")
    print(f"FC  (Factual Correct): {fc:.3f}")

# -------- ADVICE --------
if adv_scores:
    rn  = sum(x[0] for x in adv_scores) / len(adv_scores)
    sc  = sum(x[1] for x in adv_scores) / len(adv_scores)
    app = sum(x[2] for x in adv_scores) / len(adv_scores)

    print("\n📌 ADVICE TASK (Semantic Evaluation):")
    print(f"RN  (Recovery Need)   : {rn:.3f}")
    print(f"SC  (Completeness)    : {sc:.3f}")
    print(f"APP (Priority Score)  : {app:.3f}")