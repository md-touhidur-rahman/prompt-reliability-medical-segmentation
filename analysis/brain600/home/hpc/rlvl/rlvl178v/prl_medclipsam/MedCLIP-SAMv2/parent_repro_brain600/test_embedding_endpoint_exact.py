import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

sys.path.insert(0, "saliency_maps")
from scripts.methods import vision_heatmap_iba

MODEL_PATH = "./saliency_maps/model"
TOKENIZER_PATH = (
    "/home/hpc/rlvl/rlvl178v/prl_medclipsam/"
    "chuhac_BiomedCLIP-vit-bert-hf"
)

DATA = Path(
    "/home/hpc/rlvl/rlvl178v/prl_medclipsam/"
    "authors_segmentation_data/data/brain_tumors/test_images"
)

H0_PATH = Path("parent_repro_brain600/highop_prompts/H0_original.json")

CASE = "7.png"
SEED = 42


class FixedTextFeatureModel(torch.nn.Module):
    def __init__(self, base_model, fixed_z):
        super().__init__()
        self.base_model = base_model
        self.fixed_z = fixed_z.detach()

        self.vision_model = base_model.vision_model
        self.text_model = base_model.text_model

    def get_image_features(self, *args, **kwargs):
        return self.base_model.get_image_features(*args, **kwargs)

    def get_text_features(self, input_ids=None, *args, **kwargs):
        if input_ids is None:
            batch_size = 1
        else:
            batch_size = input_ids.shape[0]

        z = self.fixed_z

        if z.ndim == 1:
            z = z.unsqueeze(0)

        if z.shape[0] == 1 and batch_size != 1:
            z = z.expand(batch_size, -1)

        return z


def reset_rng():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


device = "cuda"

print("=" * 100)
print("EXACT ENDPOINT CONTROL")
print("=" * 100)

reset_rng()

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True
)

h0 = json.load(open(H0_PATH))
text = h0[CASE]

image = Image.open(DATA / CASE).convert("RGB")

image_feat = processor(
    images=image,
    return_tensors="pt"
)["pixel_values"].to(device)

text_ids = torch.tensor(
    [tokenizer.encode(text, add_special_tokens=True)],
    dtype=torch.long,
    device=device
)

# ----------------------------------------------------------
# Encode exact projected H0 feature.
# ----------------------------------------------------------

with torch.no_grad():
    z0 = model.get_text_features(text_ids).detach()

print("CASE =", CASE)
print("TEXT =", text)
print("z0 shape =", tuple(z0.shape))
print("z0 norm =", z0.norm(dim=-1).item())

# ----------------------------------------------------------
# A. ORIGINAL MODEL
#
# Reset RNG immediately before saliency generation.
# ----------------------------------------------------------

reset_rng()

original = vision_heatmap_iba(
    text_ids,
    image_feat,
    model,
    7,
    0.1,
    1.0,
    ensemble=False,
    progbar=False
)

# ----------------------------------------------------------
# B. FIXED FEATURE MODEL
#
# IMPORTANT: reset to exactly the same RNG state.
# ----------------------------------------------------------

fixed_model = FixedTextFeatureModel(model, z0)

dummy_ids = torch.tensor(
    [[0]],
    dtype=torch.long,
    device=device
)

reset_rng()

fixed = vision_heatmap_iba(
    dummy_ids,
    image_feat,
    fixed_model,
    7,
    0.1,
    1.0,
    ensemble=False,
    progbar=False
)

original = np.asarray(original, dtype=np.float64)
fixed = np.asarray(fixed, dtype=np.float64)

d = original - fixed

print()
print("=" * 100)
print("ORIGINAL vs FIXED-TEXT ENDPOINT")
print("=" * 100)

print("shape original =", original.shape)
print("shape fixed    =", fixed.shape)

print("original min/max =", original.min(), original.max())
print("fixed min/max    =", fixed.min(), fixed.max())

print("max abs diff  =", np.abs(d).max())
print("mean abs diff =", np.abs(d).mean())
print("RMSE          =", np.sqrt(np.mean(d*d)))

if original.std() > 0 and fixed.std() > 0:
    print(
        "correlation   =",
        np.corrcoef(original.ravel(), fixed.ravel())[0,1]
    )

print("allclose 1e-7 =", np.allclose(original, fixed, atol=1e-7))
print("allclose 1e-6 =", np.allclose(original, fixed, atol=1e-6))
print("allclose 1e-5 =", np.allclose(original, fixed, atol=1e-5))

# Save for inspection.
OUT = Path("parent_repro_brain600/embedding_endpoint_control")
OUT.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(OUT / "original_H0.png"),
    np.clip(original * 255, 0, 255).astype(np.uint8)
)

cv2.imwrite(
    str(OUT / "fixed_H0.png"),
    np.clip(fixed * 255, 0, 255).astype(np.uint8)
)

print()
print("Saved =", OUT)
print("=" * 100)
