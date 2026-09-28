#!/bin/bash
#SBATCH --job-name=b113_prompt_int
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=breast113_prompt_interface_%j.out
#SBATCH --error=breast113_prompt_interface_%j.err

set -euo pipefail

ROOT=/home/hpc/rlvl/rlvl178v/prl_medclipsam
V2=$ROOT/MedCLIP-SAMv2
PY=$HOME/.conda/envs/medclipsam/bin/python

DATA=$ROOT/authors_segmentation_data/data/breast_tumors/test_images
BASE=$V2/results_official/breast113_v2_B0B5_1824432
OUT=$V2/prompt_interface_intervention/breast113

SAM_SCRIPT=$V2/prompt_interface_intervention/frozen/prompt_sam_frozen.py
CKPT=$V2/segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth

SEED=20260928
EXPECTED_HASH=3101e4d13192c8efe12780b587dec1d1b2b92c26ffe13d4b81479b5022db473c

mkdir -p "$OUT"

echo "===================================================================================================="
echo " BREAST113 — FROZEN PROMPT-INTERFACE INTERVENTION"
echo "===================================================================================================="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Seed: $SEED"
echo

# ---------------------------------------------------------------------
# Verify frozen implementation before doing anything.
# ---------------------------------------------------------------------

ACTUAL_HASH=$(sha256sum "$SAM_SCRIPT" | awk '{print $1}')

echo "Frozen SAM script:"
echo "$SAM_SCRIPT"
echo "SHA256=$ACTUAL_HASH"

if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
    echo "ERROR: frozen prompt_sam hash mismatch"
    exit 1
fi

echo "HASH VERIFIED."
echo

NIMG=$(find "$DATA" -maxdepth 1 -type f | wc -l)

if [[ "$NIMG" -ne 113 ]]; then
    echo "ERROR: expected 113 input images, found $NIMG"
    exit 1
fi

echo "Input images = $NIMG"

# ---------------------------------------------------------------------
# Run one intervention condition.
#
# Interface definitions:
#
# points_pos : 10 positive points
# points_pm  : 10 positive + 10 negative points
# both_pos   : box + 10 positive points
# both_pm    : box + 10 positive + 10 negative points
# ---------------------------------------------------------------------

run_condition () {

    B="$1"
    INTERFACE="$2"

    COARSE="$BASE/$B/coarse"
    DEST="$OUT/$B/$INTERFACE"

    if [[ ! -d "$COARSE" ]]; then
        echo "ERROR: missing coarse directory: $COARSE"
        exit 1
    fi

    NCOARSE=$(find "$COARSE" -maxdepth 1 -type f | wc -l)

    if [[ "$NCOARSE" -ne 113 ]]; then
        echo "ERROR: $B has $NCOARSE coarse masks; expected 113"
        exit 1
    fi

    rm -rf "$DEST"
    mkdir -p "$DEST"

    echo
    echo "----------------------------------------------------------------------------------------------------"
    echo "$B / $INTERFACE"
    echo "----------------------------------------------------------------------------------------------------"
    echo "Coarse: $COARSE"
    echo "Output: $DEST"

    case "$INTERFACE" in

        points_pos)
            "$PY" "$SAM_SCRIPT" \
                --input "$DATA" \
                --mask-input "$COARSE" \
                --output "$DEST" \
                --model-type vit_h \
                --checkpoint "$CKPT" \
                --prompts points \
                --num-points 10 \
                --seed "$SEED"
            ;;

        points_pm)
            "$PY" "$SAM_SCRIPT" \
                --input "$DATA" \
                --mask-input "$COARSE" \
                --output "$DEST" \
                --model-type vit_h \
                --checkpoint "$CKPT" \
                --prompts points \
                --num-points 10 \
                --negative \
                --neg-num-points 10 \
                --seed "$SEED"
            ;;

        both_pos)
            "$PY" "$SAM_SCRIPT" \
                --input "$DATA" \
                --mask-input "$COARSE" \
                --output "$DEST" \
                --model-type vit_h \
                --checkpoint "$CKPT" \
                --prompts both \
                --num-points 10 \
                --seed "$SEED"
            ;;

        both_pm)
            "$PY" "$SAM_SCRIPT" \
                --input "$DATA" \
                --mask-input "$COARSE" \
                --output "$DEST" \
                --model-type vit_h \
                --checkpoint "$CKPT" \
                --prompts both \
                --num-points 10 \
                --negative \
                --neg-num-points 10 \
                --seed "$SEED"
            ;;

        *)
            echo "ERROR: unknown interface $INTERFACE"
            exit 1
            ;;
    esac

    NOUT=$(find "$DEST" -maxdepth 1 -type f | wc -l)

    echo "Generated masks = $NOUT"

    if [[ "$NOUT" -ne 113 ]]; then
        echo "ERROR: expected 113 outputs, got $NOUT"
        exit 1
    fi
}


# ---------------------------------------------------------------------
# Execute frozen 6 × 4 intervention matrix.
# Existing BOX B0-B5 outputs are deliberately NOT recomputed.
# ---------------------------------------------------------------------

for B in B0 B1 B2 B3 B4 B5; do

    run_condition "$B" points_pos
    run_condition "$B" points_pm
    run_condition "$B" both_pos
    run_condition "$B" both_pm

done


# ---------------------------------------------------------------------
# Final completeness audit
# ---------------------------------------------------------------------

echo
echo "===================================================================================================="
echo " FINAL OUTPUT AUDIT"
echo "===================================================================================================="

TOTAL=0

for B in B0 B1 B2 B3 B4 B5; do
    for I in points_pos points_pm both_pos both_pm; do

        D="$OUT/$B/$I"
        N=$(find "$D" -maxdepth 1 -type f | wc -l)

        printf "%-3s %-12s %3d\n" "$B" "$I" "$N"

        if [[ "$N" -ne 113 ]]; then
            echo "ERROR: incomplete condition $B/$I"
            exit 1
        fi

        TOTAL=$((TOTAL + N))
    done
done

echo
echo "New intervention masks = $TOTAL"

if [[ "$TOTAL" -ne 2712 ]]; then
    echo "ERROR: expected 2712 intervention masks"
    exit 1
fi

echo
echo "PASS: complete 24-condition intervention matrix."
echo
echo "Frozen implementation hash:"
sha256sum "$SAM_SCRIPT"

echo
echo "===================================================================================================="
echo " COMPLETE"
echo "===================================================================================================="
echo "End: $(date)"
