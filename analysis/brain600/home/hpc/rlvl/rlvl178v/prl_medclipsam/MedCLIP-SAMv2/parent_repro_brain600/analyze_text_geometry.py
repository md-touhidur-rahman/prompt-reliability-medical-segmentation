import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoModel, AutoTokenizer

ROOT = Path("parent_repro_brain600")

PROMPTS = {
    "H0": ROOT / "highop_prompts/H0_original.json",
    "H1": ROOT / "highop_prompts/H1_verb.json",
    "H2": ROOT / "highop_prompts/H2_intro.json",
    "L3": ROOT / "language_ladder_prompts/L3_semantic.json",
    "L4": ROOT / "language_ladder_prompts/L4_concise.json",
    "L5": ROOT / "language_ladder_prompts/L5_broad.json",
}

MODEL_PATH = "./saliency_maps/model"
TOKENIZER_PATH = (
    "/home/hpc/rlvl/rlvl178v/prl_medclipsam/"
    "chuhac_BiomedCLIP-vit-bert-hf"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", DEVICE)
print("Loading exact finetuned V2 model...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
).to(DEVICE).eval()

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True
)

data = {k: json.load(open(v)) for k, v in PROMPTS.items()}
keys = sorted(
    set.intersection(*(set(x.keys()) for x in data.values())),
    key=lambda x: int(Path(x).stem)
)

print("N images =", len(keys))

# ------------------------------------------------------------
# Extract the exact projected text representation used by IBA
# ------------------------------------------------------------

embeddings = {}
token_counts = {}

with torch.no_grad():
    for cond in PROMPTS:
        print("Extracting", cond)

        vecs = []
        counts = []

        for key in keys:
            text = data[cond][key]

            ids = tokenizer.encode(
                text,
                add_special_tokens=True
            )

            x = torch.tensor(
                [ids],
                dtype=torch.long,
                device=DEVICE
            )

            z = model.get_text_features(input_ids=x)

            vecs.append(z.squeeze(0).cpu().numpy())
            counts.append(len(ids))

        embeddings[cond] = np.stack(vecs)
        token_counts[cond] = np.asarray(counts)

# Save raw embeddings for later analyses
outdir = ROOT / "text_geometry"
outdir.mkdir(exist_ok=True)

np.savez(
    outdir / "projected_text_embeddings.npz",
    keys=np.asarray(keys),
    **embeddings
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize(x):
    return x / np.clip(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12,
        None
    )

norm = {k: normalize(v) for k, v in embeddings.items()}

# ------------------------------------------------------------
# 1. Paired trajectory relative to H0
# ------------------------------------------------------------

rows = []

for cond in PROMPTS:
    cos = np.sum(norm["H0"] * norm[cond], axis=1)

    for i, key in enumerate(keys):
        rows.append({
            "image": key,
            "condition": cond,
            "cosine_to_H0": float(cos[i]),
            "cosine_distance_to_H0": float(1.0 - cos[i]),
            "l2_to_H0": float(
                np.linalg.norm(
                    embeddings["H0"][i] - embeddings[cond][i]
                )
            ),
            "token_count": int(token_counts[cond][i]),
        })

paired = pd.DataFrame(rows)
paired.to_csv(
    outdir / "paired_embedding_trajectory.csv",
    index=False
)

print("\n===== PAIRED EMBEDDING TRAJECTORY =====")
print(
    paired.groupby("condition")[
        ["cosine_to_H0", "cosine_distance_to_H0",
         "l2_to_H0", "token_count"]
    ].agg(["mean", "std", "median"])
)

# ------------------------------------------------------------
# 2. Adjacent-step distances
# ------------------------------------------------------------

order = ["H0", "H1", "H2", "L3", "L4", "L5"]

adj_rows = []

for a, b in zip(order[:-1], order[1:]):
    cos = np.sum(norm[a] * norm[b], axis=1)

    for i, key in enumerate(keys):
        adj_rows.append({
            "image": key,
            "from": a,
            "to": b,
            "transition": f"{a}->{b}",
            "cosine_similarity": float(cos[i]),
            "cosine_distance": float(1.0 - cos[i]),
            "l2_distance": float(
                np.linalg.norm(
                    embeddings[a][i] - embeddings[b][i]
                )
            ),
        })

adj = pd.DataFrame(adj_rows)
adj.to_csv(
    outdir / "adjacent_embedding_steps.csv",
    index=False
)

print("\n===== ADJACENT LANGUAGE STEPS =====")
print(
    adj.groupby("transition")[
        ["cosine_similarity", "cosine_distance", "l2_distance"]
    ].agg(["mean", "std", "median"])
)

# ------------------------------------------------------------
# 3. Within-condition representational diversity
# ------------------------------------------------------------

div_rows = []

for cond in order:
    X = norm[cond]

    sim = X @ X.T

    iu = np.triu_indices(len(X), k=1)
    vals = sim[iu]

    div_rows.append({
        "condition": cond,
        "mean_pairwise_cosine": float(vals.mean()),
        "std_pairwise_cosine": float(vals.std()),
        "mean_pairwise_cosine_distance": float((1.0 - vals).mean()),
        "unique_prompts": len(set(data[cond].values())),
    })

div = pd.DataFrame(div_rows)
div.to_csv(
    outdir / "condition_embedding_diversity.csv",
    index=False
)

print("\n===== REPRESENTATIONAL DIVERSITY =====")
print(div.to_string(index=False))

# ------------------------------------------------------------
# 4. Exact embedding collapse count
# ------------------------------------------------------------

print("\n===== UNIQUE EMBEDDING COUNTS =====")

for cond in order:
    # same prompt should deterministically yield same embedding;
    # round only to avoid tiny numerical representation noise.
    X = np.round(embeddings[cond], decimals=6)
    n_unique = np.unique(X, axis=0).shape[0]

    print(
        f"{cond}: "
        f"{n_unique} unique embeddings / "
        f"{len(set(data[cond].values()))} unique prompts"
    )

# ------------------------------------------------------------
# 5. Per-image full trajectory
# ------------------------------------------------------------

trajectory_rows = []

for i, key in enumerate(keys):
    row = {"image": key}

    for cond in order:
        row[f"{cond}_prompt"] = data[cond][key]
        row[f"{cond}_tokens"] = int(token_counts[cond][i])

    for a, b in zip(order[:-1], order[1:]):
        row[f"cos_{a}_{b}"] = float(
            np.dot(norm[a][i], norm[b][i])
        )

    row["cos_H0_L5"] = float(
        np.dot(norm["H0"][i], norm["L5"][i])
    )

    trajectory_rows.append(row)

traj = pd.DataFrame(trajectory_rows)
traj.to_csv(
    outdir / "per_image_language_trajectory.csv",
    index=False
)

print("\nSaved:")
for p in sorted(outdir.iterdir()):
    print(" ", p)

print("\nDONE")
