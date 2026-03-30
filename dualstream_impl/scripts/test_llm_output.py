# ==========================================================
# TEST: FULL PIPELINE WITH LLM OUTPUT
# ==========================================================

import sys
import os
import torch
import open_clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.projector import MultimodalProjector


# ==========================
# CONFIG
# ==========================

PRE_IMAGE = "bata_explosion_pre_0.png"
POST_IMAGE = "bata_explosion_post_0.png"
POST_TYPE = "optical"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# LOAD CLIP
# ==========================

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()

print("✅ CLIP loaded")


# ==========================
# LOAD IMAGES
# ==========================

pre_img = preprocess(Image.open(PRE_IMAGE)).unsqueeze(0).to(device)
post_img = preprocess(Image.open(POST_IMAGE)).unsqueeze(0).to(device)

print("✅ Images loaded")


# ==========================
# ENCODER
# ==========================

with torch.no_grad():
    pre_feat = clip_model.encode_image(pre_img)
    post_feat = clip_model.encode_image(post_img)

fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

print("Fusion feature:", fusion_feat.shape)


# ==========================
# PROJECTOR
# ==========================

projector = MultimodalProjector().to(device)

proj_feat = projector(fusion_feat)

print("Projected feature:", proj_feat.shape)


# ==========================
# LOAD SMALL LLM (FOR TEST)
# ==========================

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
llm = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).to(device)

print("✅ LLM loaded")


# ==========================
# PROMPT
# ==========================

prompt = """
Please describe a comprehensive damage situation based on pre- and post-disaster images.

Give output strictly in format:

DISASTER:
BUILDING:
ROAD:
VEGETATION:
WATER_BODY:
AGRICULTURE:
CONCLUSION:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)


# ==========================
# GENERATE OUTPUT
# ==========================

output = llm.generate(
    **inputs,
    max_new_tokens=200
)

text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n================ OUTPUT ================")
print(text)