# Prompt Reliability in Language-Guided Medical Image Segmentation

[![Reproducibility](https://img.shields.io/badge/reproducibility-frozen-success)](#reproducibility-and-provenance)
[![Data](https://img.shields.io/badge/raw%20medical%20data-not%20redistributed-lightgrey)](#data-and-model-availability)

This repository contains the reproducibility materials, frozen analysis outputs, figure source data, and implementation provenance for our study of **case-level prompt reliability in language-guided medical image segmentation**.

## Overview

Language-conditioned segmentation systems are commonly summarized using dataset-level performance metrics. Such averages, however, do not directly describe whether the segmentation of an individual image remains stable when the language input is changed.

This project studies that distinction explicitly.

Rather than asking only:

> **How accurate is the model on average?**

we additionally ask:

> **For the same medical image, how much can the predicted segmentation change under controlled variation of the language prompt?**

The study examines:

1. case-level segmentation variability under language variation;
2. the relative contributions of image identity, prompt identity, and image-by-prompt interaction;
3. controlled language-perturbation mechanisms;
4. ground-truth-free detection of vulnerable cases using output disagreement;
5. frozen development/held-out evaluation;
6. robustness to prompt omission;
7. prompt-interface interventions; and
8. a cross-architecture boundary condition using Text3DSAM.

The central reliability quantity throughout the repository is **within-case variation**: multiple prompt conditions are evaluated on the same image and their resulting segmentations are compared at the case level.

---

## Study Cohorts

The frozen manuscript artifacts currently contain three principal evaluation settings:

| Cohort | Cases | Conditions | Primary role |
|---|---:|---:|---|
| Breast113 | 113 | 6 | Language-conditioned B0–B5 segmentation and GT-free vulnerability analysis |
| Brain600 | 600 | 6 | Prompt/localization perturbation and case-level reliability analysis |
| Text3DSAM AMOS liver | 30 | 4 | Cross-architecture language-prompt robustness boundary condition |

The repository also contains development/held-out evaluation artifacts and mechanistic analyses derived from these experiments.

**Important:** cohort names such as `Breast113` and `Brain600` refer to the frozen experimental cohorts used in this study. Detailed dataset provenance, selection criteria, and source citations should be read together with the manuscript and protocol documentation; they should not be inferred from the cohort labels alone.

---

## Experimental Principle

For a case \(i\) evaluated under prompt conditions \(p\), let

\[
D_{i,p}
\]

denote segmentation Dice.

A simple case-level sensitivity statistic is the within-case Dice range:

\[
R_i = \max_p D_{i,p} - \min_p D_{i,p}.
\]

This quantity asks how strongly evaluation of the **same image** changes across language conditions.

The repository additionally contains analyses based on mask disagreement, localization behavior, variance decomposition, controlled prompt trajectories, held-out vulnerability detection, and prompt-interface changes.

Dataset-level performance and case-level reliability are therefore treated as related but distinct evaluation targets.

---

## Controlled Brain600 Prompt Experiment

The frozen controlled-prompt manifest specifies:

- **600 images**
- **5 controlled same-class author-prompt conditions** in addition to the baseline configuration
- **3000 controlled image-prompt evaluations**
- finetuned BiomedCLIP
- `vvar = 0.3`
- `vbeta = 2.0`
- `vlayer = 9`
- seed `12`
- `kmeans + filter` post-processing
- SAM `vit_h`
- box prompting

The exact frozen prompt resources are preserved under [`protocols/`](protocols/).

The experiment manifest is:

[`protocols/experiment_manifest.json`](protocols/experiment_manifest.json)

and the controlled prompt panel is:

[`protocols/controlled_prompt_panel.json`](protocols/controlled_prompt_panel.json)

The corresponding brain and breast testing prompt files are also preserved in that directory.

---

## Repository Structure

```text
.
├── README.md
├── analysis/
│   ├── brain600/
│   ├── breast113/
│   ├── prompt_interface/
│   └── prompt_sensitivity/
│
├── implementation_snapshot/
│   └── ... exact modified implementation files
│
├── manuscript/
│   └── paper_v2/
│       ├── build_paper_artifacts.py
│       ├── build_publication_figures.py
│       ├── source_data/
│       ├── figure_source_data/
│       ├── publication_figure_source_data/
│       ├── figures/
│       ├── publication_figures/
│       ├── tables/
│       └── provenance/
│
├── protocols/
│   ├── brain_tumors_testing.json
│   ├── breast_tumors_testing.json
│   ├── controlled_prompt_panel.json
│   └── experiment_manifest.json
│
└── provenance/
    ├── MEDCLIPSAMV2_BASE_COMMIT.txt
    ├── MEDCLIPSAMV2_IMPLEMENTATION_DIFF.patch
    ├── MEDCLIPSAMV2_WORKTREE_STATUS.txt
    ├── PUBLICATION_REPOSITORY_SHA256.txt
    └── UPSTREAM_MEDCLIPSAMV2.txt
```

### `analysis/`

Frozen analysis scripts, result tables, evaluation records, and supporting artifacts for the principal experiments.

### `manuscript/paper_v2/source_data/`

Frozen numerical inputs used to construct the manuscript figures and tables.

These files are separated from the original model-inference outputs so that manuscript artifact generation can be audited independently of model inference.

### `manuscript/paper_v2/figure_source_data/`

Panel-level numerical source data used for manuscript figures.

### `manuscript/paper_v2/publication_figures/`

Publication-formatted versions of the manuscript figures.

### `manuscript/paper_v2/tables/`

Machine-readable CSV versions of the manuscript tables.

### `protocols/`

Frozen prompt definitions and experiment manifests.

### `implementation_snapshot/`

Copies of modified implementation files used during the experimental work. These are retained to make the computational state auditable.

### `provenance/`

Repository-level provenance, including the upstream MedCLIP-SAMv2 base commit, implementation diff, original worktree status, and SHA256 freeze manifest.

---

## Manuscript Artifact Generation

The repository separates **model inference** from **manuscript artifact generation**.

The frozen manuscript inputs are located in:

```text
manuscript/paper_v2/source_data/
```

Figures and tables are generated from these frozen result tables rather than by rerunning the segmentation models.

The principal artifact-generation scripts are:

```text
manuscript/paper_v2/build_paper_artifacts.py
manuscript/paper_v2/build_publication_figures.py
```

The scientific Python stack used for the frozen artifact build includes:

```text
Python
pandas
numpy
matplotlib
```

To regenerate the manuscript artifacts:

```bash
python manuscript/paper_v2/build_paper_artifacts.py
python manuscript/paper_v2/build_publication_figures.py
```

These commands regenerate manuscript artifacts from the frozen source data.

**They do not perform model inference.**

---

## Figures and Source Data

The repository includes both rendered figures and their underlying numerical source data.

The current publication figure set contains:

```text
Fig1_breast113_language_instability_FINAL
Fig2_frozen_detector_validation_FINAL
Fig3_brain600_perturbation_sensitivity_FINAL
Fig4_localization_decomposition_FINAL
Fig5_language_mechanism_FINAL
Fig6_cross_architecture_boundary_FINAL
FigS1_prompt_interface_tradeoff_FINAL
```

Publication-ready figures are available under:

```text
manuscript/paper_v2/publication_figures/
```

Associated numerical panel data are preserved under:

```text
manuscript/paper_v2/figure_source_data/
manuscript/paper_v2/publication_figure_source_data/
```

This separation is intentional: reported visualizations can be checked against their numerical source without rerunning the models.

---

## Tables

Frozen manuscript tables are located under:

```text
manuscript/paper_v2/tables/
```

The current table set includes:

```text
Table1_study_overview.csv
Table2_breast_heldout_detector.csv
Table3_brain600_perturbation_summary.csv
Table4_localization_decomposition.csv
TableS1_prompt_interface_vulnerability_stratified.csv
TableS2_prompt_interface_transitions.csv
TableS3_text3dsam_bootstrap.csv
```

The tables are stored as CSV so that reported values remain machine-readable and independently auditable.

---

## Reproducibility and Provenance

Reproducibility was treated as part of the experimental design rather than only as a post-hoc repository task.

The repository preserves:

- frozen source tables used for manuscript artifact generation;
- development and held-out evaluation records;
- exact prompt definitions;
- experiment manifests;
- modified implementation files;
- the upstream MedCLIP-SAMv2 base commit;
- the implementation diff relative to that checkout;
- the original worktree status; and
- SHA256 hashes for the frozen publication repository contents.

Repository-level hashes are recorded in:

```text
provenance/PUBLICATION_REPOSITORY_SHA256.txt
```

The MedCLIP-SAMv2 provenance records are:

```text
provenance/MEDCLIPSAMV2_BASE_COMMIT.txt
provenance/MEDCLIPSAMV2_IMPLEMENTATION_DIFF.patch
provenance/MEDCLIPSAMV2_WORKTREE_STATUS.txt
provenance/UPSTREAM_MEDCLIPSAMV2.txt
```

These records are intended to distinguish the upstream implementation from the experimental modifications used in this study.

---

## Frozen Evaluation Philosophy

Where development/held-out experiments are reported, the evaluation discipline is:

1. define the development/held-out partition;
2. select and freeze detector specifications using development data only;
3. evaluate the frozen specification on held-out data; and
4. do not retune the detector using held-out outcomes.

Corresponding split and evaluation records are retained in the analysis and manuscript source-data directories.

Readers should use the frozen records themselves as the authoritative source for the exact specification of each experiment.

---

## Data and Model Availability

### Included

This repository includes derived, non-image research artifacts required to audit the reported analyses, including:

- analysis code;
- prompt definitions;
- experiment manifests;
- case-level and aggregate numerical result tables;
- frozen held-out evaluation outputs;
- figure source data;
- manuscript tables;
- rendered publication figures;
- implementation diffs; and
- provenance hashes.

### Not Included

The repository intentionally does **not** redistribute:

- raw medical images;
- source medical datasets;
- large model checkpoints;
- model-weight files;
- generated bulk mask directories; or
- local compute caches.

Access to original datasets and externally developed model weights remains subject to their respective providers, licenses, terms, and access procedures.

This distinction is deliberate: the repository is a reproducibility record of the analyses and derived results, not a redistribution channel for third-party medical data or model weights.

---

## Upstream Implementation

The principal implementation provenance is tied to MedCLIP-SAMv2.

The exact upstream base commit used for the preserved experimental state is recorded in:

```text
provenance/MEDCLIPSAMV2_BASE_COMMIT.txt
```

The repository additionally stores the relevant implementation snapshot and the diff from that base so that experimental modifications can be inspected without relying on an undocumented local checkout.

---

## Reproducing the Manuscript Layer

For readers interested only in verifying manuscript figures and tables, model inference is unnecessary.

A minimal workflow is:

```bash
git clone https://github.com/md-touhidur-rahman/prompt-reliability-medical-segmentation.git
cd prompt-reliability-medical-segmentation

python manuscript/paper_v2/build_paper_artifacts.py
python manuscript/paper_v2/build_publication_figures.py
```

The scripts read the frozen files under:

```text
manuscript/paper_v2/source_data/
```

and produce the corresponding manuscript artifacts.

Full end-to-end model inference additionally requires the original datasets, appropriate model weights, and the software dependencies of the upstream segmentation pipelines. Those resources are intentionally not bundled here.

---

## Scope and Interpretation

This repository supports an empirical study of **prompt sensitivity and case-level reliability** in the evaluated language-guided medical segmentation settings.

The repository should not be interpreted as establishing that:

- every language-guided segmentation architecture exhibits the same sensitivity;
- output disagreement is a universally calibrated uncertainty measure;
- a particular alternative prompt interface universally resolves prompt sensitivity; or
- the included retrospective experiments alone establish prospective clinical utility.

The cross-architecture and prompt-interface experiments are retained partly to make these boundary conditions explicit.

---

## Repository Status

This repository was created as a frozen publication and reproducibility snapshot of the experimental work.

Subsequent documentation or manuscript-facing commits should not silently alter frozen experimental results.

Any correction to numerical results should be documented explicitly and accompanied by updated provenance.

---

## Citation

A formal citation will be added when the associated manuscript or preprint becomes available.

Until then, if you use materials from this repository, please cite the repository together with the corresponding commit hash.

```text
Md Touhidur Rahman.
Prompt Reliability in Language-Guided Medical Image Segmentation.
Reproducibility repository, 2026.
```

---

## Contact

**Md Touhidur Rahman**

For questions concerning the analyses or reproducibility materials, please use the GitHub issue tracker or the contact information provided with the associated manuscript.
