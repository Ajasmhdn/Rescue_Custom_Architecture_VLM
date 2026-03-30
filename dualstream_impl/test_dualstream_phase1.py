import torch
import open_clip
from PIL import Image
import os

# -----------------------------
# CONFIG (CHANGE PATHS HERE)
# -----------------------------
PRE_IMAGE = "../turkey_earthquake4_pre_733.png"
POST_IMAGE = "../turkey_earthquake4_sar_post_733.png"
POST_TYPE = "optical"   # change to "sar" later to test


# -----------------------------
# LOAD CLIP MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)

model = model.to(device)
model.eval()

print("✅ CLIP loaded")

# -----------------------------
# LOAD IMAGES
# -----------------------------
pre_img = preprocess(Image.open(PRE_IMAGE)).unsqueeze(0).to(device)
post_img = preprocess(Image.open(POST_IMAGE)).unsqueeze(0).to(device)

print("✅ Images loaded")

# -----------------------------
# DUAL STREAM FORWARD
# -----------------------------
with torch.no_grad():

    # optical encoder (pre always optical)
    pre_feat = model.encode_image(pre_img)

    # choose encoder based on type
    if POST_TYPE.lower() == "sar":
        print("👉 Using SAR encoder")
        post_feat = model.encode_image(post_img)
    else:
        print("👉 Using Optical encoder")
        post_feat = model.encode_image(post_img)

# -----------------------------
# FUSION (simple concat first)
# -----------------------------
fusion_feat = torch.cat([pre_feat, post_feat], dim=-1)

print("✅ Forward pass success")
print("Pre feature shape:", pre_feat.shape)
print("Post feature shape:", post_feat.shape)
print("Fusion feature shape:", fusion_feat.shape)