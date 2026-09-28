import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("parent_repro_brain600/embedding_intervention_full")
OUTDIR = Path("parent_repro_brain600/embedding_intervention_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]


def load_mask(path):
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise RuntimeError(f"Could not read {path}")
    return x > 127


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    den = a.sum() + b.sum()
    if den == 0:
        return 1.0

    return 2.0 * np.logical_and(a, b).sum() / den


# ------------------------------------------------------------------
# Verify all alpha directories
# ------------------------------------------------------------------

sets = {}

print("=" * 100)
print("MEDCLIP-SAMV2 — EMBEDDING INTERVENTION TRAJECTORY ANALYSIS")
print("=" * 100)

for alpha in ALPHAS:
    d = ROOT / f"a{alpha:.2f}"
    files = {p.name for p in d.glob("*.png")}
    sets[alpha] = files
    print(f"alpha={alpha:.2f}: {len(files)}")

common = set.intersection(*sets.values())

print(f"Common images = {len(common)}")

if len(common) != 600:
    raise RuntimeError(
        f"Expected 600 common images, found {len(common)}"
    )

for alpha in ALPHAS:
    if sets[alpha] != common:
        raise RuntimeError(
            f"Filename mismatch for alpha={alpha:.2f}"
        )


# ------------------------------------------------------------------
# Per-image trajectory
#
# D_H0(alpha) = Dice(M_alpha, M_0)
# D_L5(alpha) = Dice(M_alpha, M_1)
#
# Distance is represented as 1-Dice.
# ------------------------------------------------------------------

rows = []

ordered = sorted(
    common,
    key=lambda x: int(Path(x).stem)
)

for image_id in ordered:

    masks = {
        alpha: load_mask(
            ROOT / f"a{alpha:.2f}" / image_id
        )
        for alpha in ALPHAS
    }

    m0 = masks[0.00]
    m1 = masks[1.00]

    endpoint_dice = dice(m0, m1)
    endpoint_disagreement = 1.0 - endpoint_dice

    from_h0 = []
    to_l5 = []

    for alpha in ALPHAS:
        from_h0.append(
            1.0 - dice(masks[alpha], m0)
        )

        to_l5.append(
            1.0 - dice(masks[alpha], m1)
        )

    # Version-proof SciPy access.
    rho_h0 = spearmanr(ALPHAS, from_h0)[0]
    rho_l5 = spearmanr(ALPHAS, to_l5)[0]

    # NaN occurs when a trajectory is exactly constant.
    if not np.isfinite(rho_h0):
        rho_h0 = 0.0

    if not np.isfinite(rho_l5):
        rho_l5 = 0.0

    # A monotonic H0 -> L5 trajectory should:
    #   increase distance from H0
    #   decrease distance to L5
    monotonic_h0 = all(
        from_h0[i + 1] >= from_h0[i] - 1e-12
        for i in range(len(from_h0) - 1)
    )

    monotonic_l5 = all(
        to_l5[i + 1] <= to_l5[i] + 1e-12
        for i in range(len(to_l5) - 1)
    )

    fully_monotonic = monotonic_h0 and monotonic_l5

    # Maximum single-step movement.
    step_dice = [
        dice(masks[ALPHAS[i]], masks[ALPHAS[i + 1]])
        for i in range(len(ALPHAS) - 1)
    ]

    step_disagreement = [
        1.0 - x for x in step_dice
    ]

    max_step = max(step_disagreement)
    max_step_idx = int(np.argmax(step_disagreement))

    # Deviation of intermediate masks from a simple linear
    # endpoint-disagreement expectation.
    #
    # If movement were perfectly proportional to alpha:
    # distance from H0 ~= alpha * endpoint distance.
    expected_from_h0 = [
        alpha * endpoint_disagreement
        for alpha in ALPHAS
    ]

    trajectory_deviation = np.mean([
        abs(obs - exp)
        for obs, exp in zip(
            from_h0,
            expected_from_h0
        )
    ])

    row = {
        "filename": image_id,
        "endpoint_dice": endpoint_dice,
        "endpoint_disagreement": endpoint_disagreement,
        "rho_from_h0": rho_h0,
        "rho_to_l5": rho_l5,
        "monotonic_from_h0": int(monotonic_h0),
        "monotonic_to_l5": int(monotonic_l5),
        "fully_monotonic": int(fully_monotonic),
        "max_step_disagreement": max_step,
        "max_step_start_alpha": ALPHAS[max_step_idx],
        "trajectory_deviation": trajectory_deviation,
    }

    for alpha, value in zip(ALPHAS, from_h0):
        row[f"dist_H0_a{alpha:.2f}"] = value

    for alpha, value in zip(ALPHAS, to_l5):
        row[f"dist_L5_a{alpha:.2f}"] = value

    for i, value in enumerate(step_disagreement):
        row[
            f"step_{ALPHAS[i]:.2f}_{ALPHAS[i+1]:.2f}"
        ] = value

    rows.append(row)


# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------

endpoint = np.array([
    r["endpoint_disagreement"] for r in rows
])

rho_h0 = np.array([
    r["rho_from_h0"] for r in rows
])

rho_l5 = np.array([
    r["rho_to_l5"] for r in rows
])

fully_monotonic = np.array([
    r["fully_monotonic"] for r in rows
], dtype=bool)

max_step = np.array([
    r["max_step_disagreement"] for r in rows
])

traj_dev = np.array([
    r["trajectory_deviation"] for r in rows
])


lines = []

def P(x=""):
    print(x)
    lines.append(str(x))


P("")
P("=" * 100)
P("ENDPOINT H0 ↔ L5 OUTPUT DISAGREEMENT")
P("=" * 100)

P(f"Mean   = {endpoint.mean():.6f}")
P(f"Median = {np.median(endpoint):.6f}")
P(f"Q25    = {np.quantile(endpoint, 0.25):.6f}")
P(f"Q75    = {np.quantile(endpoint, 0.75):.6f}")
P(f"P90    = {np.quantile(endpoint, 0.90):.6f}")
P(f"Max    = {endpoint.max():.6f}")


P("")
P("=" * 100)
P("MONOTONICITY OF EMBEDDING INTERVENTION")
P("=" * 100)

n = len(rows)

mh0 = sum(
    r["monotonic_from_h0"] for r in rows
)

ml5 = sum(
    r["monotonic_to_l5"] for r in rows
)

mf = sum(
    r["fully_monotonic"] for r in rows
)

P(
    f"Monotonic away from H0 = "
    f"{mh0}/{n} = {100*mh0/n:.2f}%"
)

P(
    f"Monotonic toward L5    = "
    f"{ml5}/{n} = {100*ml5/n:.2f}%"
)

P(
    f"Fully monotonic        = "
    f"{mf}/{n} = {100*mf/n:.2f}%"
)

P("")
P(
    f"Mean Spearman rho, distance from H0 = "
    f"{rho_h0.mean():+.4f}"
)

P(
    f"Mean Spearman rho, distance to L5   = "
    f"{rho_l5.mean():+.4f}"
)


P("")
P("=" * 100)
P("STEPWISE MOVEMENT")
P("=" * 100)

for i in range(len(ALPHAS) - 1):

    key = (
        f"step_{ALPHAS[i]:.2f}_"
        f"{ALPHAS[i+1]:.2f}"
    )

    vals = np.array([
        r[key] for r in rows
    ])

    P(
        f"{ALPHAS[i]:.2f} -> {ALPHAS[i+1]:.2f}: "
        f"mean={vals.mean():.6f} "
        f"median={np.median(vals):.6f} "
        f"P90={np.quantile(vals,0.90):.6f}"
    )


P("")
P("=" * 100)
P("TRAJECTORY NONLINEARITY")
P("=" * 100)

P(
    f"Mean trajectory deviation   = "
    f"{traj_dev.mean():.6f}"
)

P(
    f"Median trajectory deviation = "
    f"{np.median(traj_dev):.6f}"
)

P(
    f"P90 trajectory deviation    = "
    f"{np.quantile(traj_dev,0.90):.6f}"
)


P("")
P("=" * 100)
P("WHERE DOES THE LARGEST OUTPUT CHANGE OCCUR?")
P("=" * 100)

for a in ALPHAS[:-1]:

    count = sum(
        abs(
            r["max_step_start_alpha"] - a
        ) < 1e-9
        for r in rows
    )

    P(
        f"{a:.2f} -> {a+0.25:.2f}: "
        f"{count:3d}/{n} = "
        f"{100*count/n:6.2f}%"
    )


P("")
P("=" * 100)
P("TOP 20 STRONGEST H0-L5 ENDPOINT CHANGES")
P("=" * 100)

top = sorted(
    rows,
    key=lambda r: r["endpoint_disagreement"],
    reverse=True
)[:20]

for i, r in enumerate(top, 1):

    P(
        f"{i:02d}. {r['filename']:<10} "
        f"| endpoint-dis={r['endpoint_disagreement']:.4f} "
        f"| rhoH0={r['rho_from_h0']:+.3f} "
        f"| rhoL5={r['rho_to_l5']:+.3f} "
        f"| mono={r['fully_monotonic']} "
        f"| max-step={r['max_step_disagreement']:.4f} "
        f"@ {r['max_step_start_alpha']:.2f}"
    )


# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------

csv_path = (
    OUTDIR /
    "embedding_intervention_trajectory.csv"
)

fields = list(rows[0].keys())

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=fields
    )
    w.writeheader()
    w.writerows(rows)


summary_path = (
    OUTDIR /
    "embedding_intervention_summary.txt"
)

with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")


print("")
print("=" * 100)
print("SAVED")
print("=" * 100)
print("CSV     =", csv_path)
print("SUMMARY =", summary_path)
print("")
print("DONE")
