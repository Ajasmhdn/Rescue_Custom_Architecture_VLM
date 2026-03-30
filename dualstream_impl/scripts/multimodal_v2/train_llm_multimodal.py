# ==========================================================
# TRAIN LLM WITH MULTIMODAL INPUT (REAL VLM)
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.dataset_loader import DisasterDataset
from models.projector import MultimodalProjector


# ==========================
# CONFIG
# ==========================

JSON_PATH = "data/train.json"
OUTPUT_DIR = "outputs/multimodal_v2/llm_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 2
LR = 2e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")


# ==========================
# LOAD CLIP
# ==========================

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()


# ==========================
# LOAD PROJECTOR
# ==========================

projector = MultimodalProjector().to(device)
projector.load_state_dict(torch.load("outputs/multimodal_v2/projector_v2.pt"))
projector.eval()

print("✅ Projector loaded")


# ==========================
# LOAD DATASET
# ==========================

dataset = DisasterDataset(JSON_PATH, preprocess)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================
# LOAD LLM
# ==========================

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16
).to(device)


# ==========================
# APPLY LORA
# ==========================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
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

        # -------------------------
        # IMAGE → FEATURES
        # -------------------------
        with torch.no_grad():
            pre_feat = clip_model.encode_image(pre_img)
            post_feat = clip_model.encode_image(post_img)

        fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

        with torch.no_grad():
            proj_feat = projector(fusion_feat)  # (B, 1536)
        proj_feat = proj_feat.to(torch.float16) 
        proj_feat = proj_feat.unsqueeze(1)  # (B, 1, 1536)

        # -------------------------
        # TEXT TOKENS
        # -------------------------
        tokens = tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # -------------------------
        # TEXT EMBEDDINGS
        # -------------------------
        text_embeds = model.get_input_embeddings()(tokens.input_ids)

        # -------------------------
        # CONCAT IMAGE + TEXT
        # -------------------------
        inputs_embeds = torch.cat([proj_feat, text_embeds], dim=1)

        # -------------------------
        # ADJUST LABELS
        # -------------------------
        labels_ids = tokens.input_ids

        pad_token = torch.full(
            (labels_ids.shape[0], 1),
            -100,
            device=device
        )

        new_labels = torch.cat([pad_token, labels_ids], dim=1)

        # -------------------------
        # FORWARD
        # -------------------------
        outputs = model(
            inputs_embeds=inputs_embeds,
            labels=new_labels
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


# ==========================
# SAVE LORA
# ==========================

model.save_pretrained(OUTPUT_DIR)

print(f"\n💾 LoRA saved → {OUTPUT_DIR}")
print("✅ TRAINING DONE")