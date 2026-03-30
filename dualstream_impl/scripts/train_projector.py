# ==========================================================
# TRAIN PROJECTOR (FINAL GPU VERSION + SAVE METRICS + WEIGHTS)
# ==========================================================

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open_clip
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# FIX PATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataset_loader import DisasterDataset
from models.projector import MultimodalProjector


# ==========================
# CONFIG
# ==========================

JSON_PATH = "data/train.json"
OUTPUT_DIR = "outputs"

BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Using device: {device}")


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
# DATASET
# ==========================

dataset = DisasterDataset(JSON_PATH, preprocess)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

print("Dataset size:", len(dataset))


# ==========================
# PROJECTOR
# ==========================

projector = MultimodalProjector().to(device)


# ==========================
# TEXT EMBEDDING MODEL
# ==========================

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

text_model.eval()
print("✅ Text encoder loaded")


# ==========================
# LOSS + OPTIMIZER
# ==========================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(projector.parameters(), lr=LR)


# ==========================
# METRICS STORAGE
# ==========================

metrics_log = []


# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(EPOCHS):

    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    total_loss = 0

    loop = tqdm(loader)

    for pre_img, post_img, labels in loop:

        pre_img = pre_img.to(device, non_blocking=True)
        post_img = post_img.to(device, non_blocking=True)

        # -------- IMAGE FEATURES --------
        with torch.no_grad():
            pre_feat = clip_model.encode_image(pre_img)
            post_feat = clip_model.encode_image(post_img)

        fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

        proj_feat = projector(fusion_feat)

        # -------- LABEL EMBEDDING --------
        tokens = tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_feat = text_model(**tokens).last_hidden_state.mean(dim=1)

        # -------- LOSS --------
        loss = loss_fn(proj_feat, text_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    epoch_loss = total_loss / len(loader)

    print(f"✅ Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    metrics_log.append({
        "epoch": epoch + 1,
        "loss": epoch_loss
    })


# ==========================
# SAVE WEIGHTS
# ==========================

weight_path = os.path.join(OUTPUT_DIR, "projector_trained.pt")
torch.save(projector.state_dict(), weight_path)

print(f"\n💾 Weights saved → {weight_path}")


# ==========================
# SAVE METRICS
# ==========================

metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics_log, f, indent=4)

print(f"📊 Metrics saved → {metrics_path}")

print("\n✅ TRAINING COMPLETE")