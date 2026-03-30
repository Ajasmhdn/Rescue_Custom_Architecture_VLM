# ==========================================================
# TEST SET EVALUATION (MULTIMODAL V2 - PRESENTATION VERSION)
# ==========================================================

import sys
import os
import json
import torch
import open_clip
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================
# FIX PATH
# ==========================
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.projector import MultimodalProjector

# ==========================
# PATH CONFIG
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

TEST_JSON = os.path.join(BASE_DIR, "data/testdata/test.json")
IMG_FOLDER = os.path.join(BASE_DIR, "data/testdata/test_images")
OUT_FILE = os.path.join(BASE_DIR, "outputs/multimodal_v2/test_predictions_v2_correctedversion.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)

# ==========================
# LOAD DATA
# ==========================

with open(TEST_JSON) as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} test samples")

# ==========================
# LOAD MODELS
# ==========================

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)
clip_model = clip_model.to(device).eval()

projector = MultimodalProjector().to(device)
projector.load_state_dict(
    torch.load("outputs/multimodal_v2/projector_v2.pt", map_location=device)
)
projector.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16
).to(device)

llm = PeftModel.from_pretrained(llm, "outputs/multimodal_v2/llm_lora")
llm.eval()

print("✅ Models loaded")

# ==========================
# PROMPTS (SOFT + EFFECTIVE)
# ==========================

CAPTION_PROMPT = """
You are a disaster assessment expert.

Analyze the pre- and post-disaster images and describe the damage.

Include:
- disaster type
- building damage
- road condition
- vegetation impact
- water bodies
- agriculture
- overall conclusion
"""

ADVICE_PROMPT = """
You are a disaster recovery expert.

Analyze the damage and suggest recovery strategies.

Include:
- immediate recovery actions
- long-term recovery strategies
"""

# ==========================
# EVALUATION LOOP
# ==========================

results = []
skipped = 0

for item in tqdm(data):

    try:
        # -------------------------
        # IMAGE PATH FIX
        # -------------------------
        pre_name = os.path.basename(item["pre_image_path"].replace("\\", "/"))
        post_name = os.path.basename(item["post_image_path"].replace("\\", "/"))

        pre_path = os.path.join(IMG_FOLDER, pre_name)
        post_path = os.path.join(IMG_FOLDER, post_name)

        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            skipped += 1
            continue

        # -------------------------
        # TASK ROUTING (FIXED)
        # -------------------------
        task_raw = item.get("task", "").lower()

        if "restoration" in task_raw:
            prompt = ADVICE_PROMPT
            task_type = "advice"
        else:
            prompt = CAPTION_PROMPT
            task_type = "caption"

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        pre_img = preprocess(Image.open(pre_path)).unsqueeze(0).to(device)
        post_img = preprocess(Image.open(post_path)).unsqueeze(0).to(device)

        # -------------------------
        # IMAGE → EMBEDDING
        # -------------------------
        with torch.no_grad():
            pre_feat = clip_model.encode_image(pre_img)
            post_feat = clip_model.encode_image(post_img)

        fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

        proj_feat = projector(fusion_feat)
        proj_feat = proj_feat.unsqueeze(1).to(torch.float16)

        # -------------------------
        # TEXT EMBEDDING
        # -------------------------
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = llm.get_input_embeddings()(tokens.input_ids)

        # -------------------------
        # MULTIMODAL FUSION
        # -------------------------
        inputs_embeds = torch.cat([proj_feat, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

        # -------------------------
        # GENERATE
        # -------------------------
        output = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False
        )

        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # avoid empty predictions
        if len(pred) == 0:
            pred = "No meaningful output generated."

        # -------------------------
        # STORE RESULT
        # -------------------------
        results.append({
            "task": task_type,
            "ground_truth": item["ground_truth"],
            "prediction": pred
        })

    except Exception as e:
        print("⚠️ Error:", e)
        skipped += 1
        continue

# ==========================
# SAVE RESULTS
# ==========================

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Saved predictions →", OUT_FILE)
print(f"⚠️ Skipped samples: {skipped}")
print(f"✅ Final usable samples: {len(results)}")