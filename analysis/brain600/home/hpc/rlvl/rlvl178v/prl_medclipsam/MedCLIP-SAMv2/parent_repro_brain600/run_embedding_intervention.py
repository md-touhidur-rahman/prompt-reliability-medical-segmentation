import argparse
import json
import os
import random
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import sys
sys.path.insert(0, "saliency_maps")
from scripts.methods import vision_heatmap_iba


ROOT = Path("parent_repro_brain600")

MODEL_PATH = "./saliency_maps/model"
TOKENIZER_PATH = (
    "/home/hpc/rlvl/rlvl178v/prl_medclipsam/"
    "chuhac_BiomedCLIP-vit-bert-hf"
)

H0_PATH = ROOT / "highop_prompts/H0_original.json"
L5_PATH = ROOT / "language_ladder_prompts/L5_broad.json"

ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]


class FixedTextFeatureModel(torch.nn.Module):
    """
    Proxy around the BiomedCLIP model.

    Vision pathway is untouched.
    Text features are replaced by a fixed intervened projected representation.
    """
    def __init__(self, base_model, fixed_z):
        super().__init__()
        self.base_model = base_model
        self.fixed_z = fixed_z.detach()

        # Required by vision_heatmap_iba / IBA machinery.
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


def encode_text(model, tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        z = model.get_text_features(input_ids=x)

    return z.detach()


def deterministic_image_seed(image_id, base_seed):
    """
    Stable per-image seed.

    All alpha values for one image receive the same seed so the
    stochastic IBA realization is matched across the trajectory.
    """
    payload = f"{base_seed}:{image_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    offset = int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False
    )
    return (int(base_seed) + offset) % (2**31 - 1)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device

    print("Loading exact finetuned V2 model...")

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
    l5 = json.load(open(L5_PATH))

    common = sorted(
        set(h0) & set(l5),
        key=lambda x: int(Path(x).stem)
    )

    if args.limit is not None:
        common = common[:args.limit]

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards - 1}]"
        )

    if args.num_shards > 1:
        common = common[
            args.shard_index::args.num_shards
        ]

    print("Images =", len(common))
    print(
        "Shard =",
        args.shard_index,
        "/",
        args.num_shards
    )
    print("Alphas =", ALPHAS)

    outroot = Path(args.output_root)
    outroot.mkdir(parents=True, exist_ok=True)

    for alpha in ALPHAS:
        (outroot / f"a{alpha:.2f}").mkdir(
            parents=True,
            exist_ok=True
        )

    for image_id in tqdm(common):

        image_path = Path(args.input_path) / image_id

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print("SKIP", image_id, e)
            continue

        image_feat = processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(device)

        # Endpoint projected representations.
        z0 = encode_text(
            model, tokenizer, h0[image_id], device
        )

        z5 = encode_text(
            model, tokenizer, l5[image_id], device
        )

        # Dummy token tensor: FixedTextFeatureModel ignores token identity,
        # but IBA uses its batch dimension.
        dummy_ids = torch.tensor(
            [[0]],
            dtype=torch.long,
            device=device
        )

        for alpha in ALPHAS:

            outfile = (
                outroot /
                f"a{alpha:.2f}" /
                image_id
            )

            if outfile.exists() and not args.overwrite:
                continue

            z = (1.0 - alpha) * z0 + alpha * z5

            intervention_model = FixedTextFeatureModel(
                model,
                z
            )

            # MATCHED STOCHASTIC CONTROL:
            # Same image -> same RNG state for every alpha.
            image_seed = deterministic_image_seed(
                image_id,
                args.seed
            )
            reset_rng(image_seed)

            vmap = vision_heatmap_iba(
                dummy_ids,
                image_feat,
                intervention_model,
                args.vlayer,
                args.vbeta,
                args.vvar,
                ensemble=False,
                progbar=False
            )

            img = np.array(image)

            vmap = cv2.resize(
                np.array(vmap),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imwrite(
                str(outfile),
                vmap * 255
            )

    print("\nDONE")
    print("Saved:", outroot)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-path",
        required=True
    )

    p.add_argument(
        "--output-root",
        default="parent_repro_brain600/"
                "embedding_intervention_saliency"
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--vlayer", type=int, default=7)
    p.add_argument("--vbeta", type=float, default=0.1)
    p.add_argument("--vvar", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--shard-index",
        type=int,
        default=0
    )

    p.add_argument(
        "--num-shards",
        type=int,
        default=1
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None
    )

    p.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = p.parse_args()
    main(args)
