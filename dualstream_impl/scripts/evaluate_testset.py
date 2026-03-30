# ==========================================================
# TEST SET EVALUATION (FINAL CLEAN VERSION)
# ==========================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import json
import torch
import open_clip
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from models.projector import MultimodalProjector

# ==========================
# PATH CONFIG
# ==========================

TEST_JSON = "data/testdata/test.json"
IMG_FOLDER = "data/testdata/test_images"
OUT_FILE = "outputs/test_predictions.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)

# ==========================
# LOAD DATA
# ==========================

with open(TEST_JSON) as f:
    data = json.load(f)

# ==========================
# LOAD MODELS
# ==========================

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)
clip_model = clip_model.to(device).eval()

projector = MultimodalProjector().to(device)
projector.load_state_dict(torch.load("outputs/projector_trained.pt", map_location=device))
projector.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    dtype=torch.float16
).to(device)

llm = PeftModel.from_pretrained(llm, "outputs/llm_lora")
llm.eval()

print("✅ Models loaded")

# ==========================
# EVALUATION LOOP
# ==========================

results = []

for item in tqdm(data):

    pre_name = os.path.basename(item["pre_image_path"].replace("\\", "/"))
    post_name = os.path.basename(item["post_image_path"].replace("\\", "/"))

    pre_path = os.path.join(IMG_FOLDER, pre_name)
    post_path = os.path.join(IMG_FOLDER, post_name)
    # prompt select
    if item["task"] == "disaster caption":
        prompt = "Please describe a comprehensive damage situation based on pre- and post-disaster images."
    else:
        prompt = "Analyze disaster impacts and propose restoration strategies addressing both immediate recovery needs and long-term resilience considerations."

    pre_img = preprocess(Image.open(pre_path)).unsqueeze(0).to(device)
    post_img = preprocess(Image.open(post_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        pre_feat = clip_model.encode_image(pre_img)
        post_feat = clip_model.encode_image(post_img)

    fusion_feat = torch.cat([pre_feat, post_feat], dim=1)
    proj_feat = projector(fusion_feat)

    tokens = tokenizer(prompt, return_tensors="pt").to(device)

    output = llm.generate(
        **tokens,
        max_new_tokens=220,
        do_sample=False
    )

    pred = tokenizer.decode(output[0], skip_special_tokens=True)

    results.append({
        "ground_truth": item["ground_truth"],
        "prediction": pred
    })

# ==========================
# SAVE
# ==========================

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("✅ Saved predictions →", OUT_FILE)