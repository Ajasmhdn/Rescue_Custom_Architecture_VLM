# ==========================================================
# TEST DATASET LOADER (FINAL WORKING VERSION)
# ==========================================================

import sys
import os

# FIX IMPORT PATH (IMPORTANT)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import open_clip
from data.dataset_loader import DisasterDataset


# ==========================
# LOAD CLIP PREPROCESS
# ==========================

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)


# ==========================
# DATASET PATH (UPDATED)
# ==========================

dataset = DisasterDataset("data/train.json", preprocess)

print("Dataset size:", len(dataset))


# ==========================
# TEST ONE SAMPLE
# ==========================

pre, post, label = dataset[0]

print("\nPre shape:", pre.shape)
print("Post shape:", post.shape)

print("\nLabel preview:\n")
print(label[:300])


print("\n✅ DATASET WORKING")