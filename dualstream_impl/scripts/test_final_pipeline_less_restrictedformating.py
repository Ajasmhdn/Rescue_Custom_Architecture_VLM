# # ==========================================================
# # FINAL PIPELINE TEST (FINAL WORKING VERSION)
# # ==========================================================

# import sys
# import os
# import torch
# import open_clip
# import re
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# from PIL import Image

# # ==========================
# # FIX PATH
# # ==========================
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from models.projector import MultimodalProjector


# # ==========================
# # CONFIG
# # ==========================
# TASK = "advice"   # change → "caption" or "advice"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"✅ Device: {device}")


# # ==========================
# # LOAD CLIP
# # ==========================
# clip_model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-B-16",
#     pretrained="openai"
# )

# clip_model = clip_model.to(device)
# clip_model.eval()
# print("✅ CLIP loaded")


# # ==========================
# # LOAD PROJECTOR
# # ==========================
# projector = MultimodalProjector().to(device)
# projector.load_state_dict(torch.load("outputs/projector_trained.pt", map_location=device))
# projector.eval()
# print("✅ Projector loaded")


# # ==========================
# # LOAD LLM + LoRA
# # ==========================
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# llm = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-1.5B-Instruct",
#     dtype=torch.float16
# ).to(device)

# llm = PeftModel.from_pretrained(llm, "outputs/llm_lora")
# llm.eval()

# print("✅ LLM + LoRA loaded")


# # ==========================
# # LOAD TEST IMAGES
# # ==========================
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# pre_path = os.path.join(BASE_DIR, "bata_explosion_pre_0.png")
# post_path = os.path.join(BASE_DIR, "bata_explosion_post_0.png")

# pre_img = preprocess(Image.open(pre_path)).unsqueeze(0).to(device)
# post_img = preprocess(Image.open(post_path)).unsqueeze(0).to(device)

# print("✅ Images loaded")


# # ==========================
# # IMAGE FEATURES
# # ==========================
# with torch.no_grad():
#     pre_feat = clip_model.encode_image(pre_img)
#     post_feat = clip_model.encode_image(post_img)

# fusion_feat = torch.cat([pre_feat, post_feat], dim=1)

# with torch.no_grad():
#     proj_feat = projector(fusion_feat).to(torch.float16)


# # ==========================
# # PROMPT
# # ==========================
# if TASK == "caption":

#     prompt = (
#         "Please describe a comprehensive damage situation based on pre- and post-disaster images."
#     )

# elif TASK == "advice":

#     prompt = (
#         "Analyze disaster impacts and propose restoration strategies addressing both immediate recovery needs and long-term resilience considerations."
#     )

# inputs = tokenizer(prompt, return_tensors="pt").to(device)


# # ==========================
# # GENERATE
# # ==========================
# with torch.no_grad():
#     output = llm.generate(
#         **inputs,
#         max_new_tokens=220,
#         do_sample=False,
#         repetition_penalty=1.15,
#         eos_token_id=tokenizer.eos_token_id
#     )

# raw_text = tokenizer.decode(output[0], skip_special_tokens=True)


# # ==========================
# # 🔥 FINAL FORMATTER
# # ==========================

# def paragraph_to_caption(text):

#     # remove prompt echo
#     text = text.replace(prompt, "").strip()

#     # remove disaster-type opening sentences
#     text = re.sub(
#         r"(the disaster[^.]*\.)|(this disaster[^.]*\.)|(an earthquake[^.]*\.)|(a hurricane[^.]*\.)",
#         "",
#         text,
#         flags=re.I
#     )

#     def find_sentence(keyword):

#         match = re.search(rf"([^.]*{keyword}[^.]*\.)", text, re.I)

#         if not match:
#             return "Not observed clearly."

#         sent = match.group(1).strip()

#         # remove disaster references inside sentence
#         sent = re.sub(r"\b(disaster|earthquake|hurricane|flood|explosion)\b[^.]*", "", sent, flags=re.I)

#         return sent.strip()

#     return f"""
# BUILDING: {find_sentence("building")}
# ROAD: {find_sentence("road")}
# VEGETATION: {find_sentence("vegetation")}
# WATER_BODY: {find_sentence("water")}
# AGRICULTURE: {find_sentence("agricultur")}
# CONCLUSION: Overall damage mainly affects built structures.
# """


# def paragraph_to_advice(text):

#     # remove prompt echo
#     text = text.replace(prompt, "").strip()

#     # remove unwanted intro parts
#     text = re.sub(
#         r"(disaster analysis[^.]*\.)|(the disaster[^.]*\.)|(this disaster[^.]*\.)",
#         "",
#         text,
#         flags=re.I
#     )

#     # remove conclusion or extra instructions
#     text = re.split(r"Conclusion:|Create|Overall", text, flags=re.I)[0]

#     # split sentences
#     sentences = re.split(r'(?<=\.)\s+', text)

#     # keep strong sentences only
#     clean_sentences = [
#         s.strip() for s in sentences
#         if len(s.split()) > 6
#     ]

#     half = len(clean_sentences) // 2

#     immediate = " ".join(clean_sentences[:half]).strip()
#     longterm = " ".join(clean_sentences[half:]).strip()

#     return f"""
# Immediate: {immediate}

# Long-term: {longterm}
# """                                                                                                                                                                                                    



# # ==========================
# # APPLY FORMATTER
# # ==========================
# if TASK == "caption":
#     final_output = paragraph_to_caption(raw_text)

# else:
#     final_output = paragraph_to_advice(raw_text)


# print("\n================ OUTPUT ================\n")
# print(final_output)