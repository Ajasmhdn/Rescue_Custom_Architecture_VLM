# ==========================================================
# FULL TEST: DUAL ENCODER + PROJECTOR (WORKING VERSION)
# ==========================================================

import sys
import os
import torch
import open_clip
from PIL import Image

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.projector import MultimodalProjector


# ==========================================================
# CONFIG (CHANGE PATHS HERE)
# ==========================================================

PRE_IMAGE = "bata_explosion_pre_0.png"
POST_IMAGE = "bata_explosion_post_0.png"

POST_TYPE = "optical"   # change to "sar" to test


# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# LOAD CLIP MODEL
# ==========================================================

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)

model = model.to(device)
model.eval()

print("✅ CLIP loaded")


# ==========================================================
# LOAD IMAGES
# ==========================================================

pre_img = preprocess(Image.open(PRE_IMAGE)).unsqueeze(0).to(device)
post_img = preprocess(Image.open(POST_IMAGE)).unsqueeze(0).to(device)

print("✅ Images loaded")


# ==========================================================
# ENCODER FORWARD PASS
# ==========================================================

with torch.no_grad():

    pre_feat = model.encode_image(pre_img)

    if POST_TYPE == "sar":
        print("👉 Using SAR encoder (currently same CLIP for test)")
        post_feat = model.encode_image(post_img)
    else:
        print("👉 Using Optical encoder")
        post_feat = model.encode_image(post_img)


print("Pre feature shape:", pre_feat.shape)
print("Post feature shape:", post_feat.shape)


# ==========================================================
# FUSION
# ==========================================================

fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

print("Fusion feature shape:", fusion_feat.shape)


# ==========================================================
# PROJECTOR
# ==========================================================

projector = MultimodalProjector().to(device)

proj_feat = projector(fusion_feat)

print("Projected feature shape:", proj_feat.shape)


print("\n✅ FULL PIPELINE WORKING")