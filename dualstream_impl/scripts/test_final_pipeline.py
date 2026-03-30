# ==========================================================
# FINAL PIPELINE (REAL MULTIMODAL - FIXED)
# ==========================================================

import sys
import os
import torch
import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image

# FIX PATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.projector import MultimodalProjector

# ==========================
# CONFIG
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

# choose task
TASK = "caption"   # "caption" or "advice"

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
# LOAD PROJECTOR (V2)
# ==========================

projector = MultimodalProjector().to(device)
projector.load_state_dict(torch.load("outputs/multimodal_v2/projector_v2.pt"))
projector.eval()

print("✅ Projector loaded")

# ==========================
# LOAD LLM + LORA
# ==========================

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16
).to(device)

llm = PeftModel.from_pretrained(llm, "outputs/llm_lora")

llm.eval()

print("✅ LLM + LoRA loaded")

# ==========================
# LOAD IMAGE
# ==========================

pre_path = "bata_explosion_pre_0.png"
post_path = "bata_explosion_post_0.png"

pre_img = preprocess(Image.open(pre_path)).unsqueeze(0).to(device)
post_img = preprocess(Image.open(post_path)).unsqueeze(0).to(device)

print("✅ Images loaded")

# ==========================
# IMAGE ENCODING
# ==========================

with torch.no_grad():
    pre_feat = clip_model.encode_image(pre_img)
    post_feat = clip_model.encode_image(post_img)

fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

proj_feat = projector(fusion_feat)   # (1, 1536)

# 👉 IMPORTANT FIX
proj_feat = proj_feat.unsqueeze(1)   # (1, 1, 1536)

# match dtype with LLM
proj_feat = proj_feat.to(torch.float16)

print("Projected feature:", proj_feat.shape)

# ==========================
# PROMPT CONTROL
# ==========================

if TASK == "caption":
    prompt = "Describe disaster damage."
elif TASK == "advice":
    prompt = "Provide recovery plan."

# tokenize
tokens = tokenizer(prompt, return_tensors="pt").to(device)

# get text embeddings
text_embeds = llm.get_input_embeddings()(tokens.input_ids)

# ==========================
# 🔥 MULTIMODAL FUSION
# ==========================

# CONCAT: [IMAGE TOKEN] + [TEXT TOKENS]
inputs_embeds = torch.cat([proj_feat, text_embeds], dim=1)

# fix attention mask
attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

# ==========================
# GENERATE
# ==========================

output = llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=200
)

print("\n================ OUTPUT ================\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))