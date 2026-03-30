# ==========================================================
# LLM ALIGNMENT TRAINING (FINAL STAGE)
# ==========================================================

import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# FIX PATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset_loader import DisasterDataset
from models.projector import MultimodalProjector


# ==========================
# CONFIG
# ==========================

JSON_PATH = "data/train.json"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Device:", device)


# ==========================
# LOAD CLIP (FROZEN)
# ==========================

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()


# ==========================
# LOAD PROJECTOR (TRAINED)
# ==========================

projector = MultimodalProjector().to(device)
projector.load_state_dict(torch.load("outputs/projector_trained.pt"))
projector.eval()

print("✅ Projector loaded")


# ==========================
# DATASET
# ==========================

dataset = DisasterDataset(JSON_PATH, preprocess)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ==========================
# LOAD LLM
# ==========================

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16
).to(device)

print("✅ LLM loaded")


# ==========================
# LORA CONFIG
# ==========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("✅ LoRA ready")


# ==========================
# OPTIMIZER
# ==========================

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(EPOCHS):

    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    loop = tqdm(loader)

    for pre_img, post_img, labels in loop:

        pre_img = pre_img.to(device)
        post_img = post_img.to(device)

        with torch.no_grad():
            pre_feat = clip_model.encode_image(pre_img)
            post_feat = clip_model.encode_image(post_img)

        fusion_feat = torch.cat([pre_feat, post_feat], dim=1)
        proj_feat = projector(fusion_feat)

        # PROMPT FORMAT
        prompts = [
            f"Describe disaster damage:\n{label}"
            for label in labels
        ]

        tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

        outputs = model(**tokens, labels=tokens["input_ids"])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


print("\n💾 Saving LoRA weights...")
model.save_pretrained("outputs/llm_lora")

print("✅ DONE")