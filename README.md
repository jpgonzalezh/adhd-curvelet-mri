# Quantifying Brain Morphological Alterations in ADHD: A Curvelet-Based Structural MRI Study

Analysis code for the structural-MRI Curvelet study of ADHD using the ADHD-200
and WMR-ADHD datasets. The pipeline characterizes regional brain morphology with
a multi-scale Curvelet representation summarized by Generalized Gaussian
parameters (α, β, μ), and evaluates supervised and unsupervised separability
between ADHD and typically developing controls (TDC) under a rigorous,
leakage-free protocol.

## Reproducibility and leakage control

All data-dependent transformations — z-score normalization, k-means centroids,
and region/parameter selection — are fit **on training folds only** and applied
unchanged to held-out data. In the supervised experiments, model and feature
selection occur inside the inner loop of a nested cross-validation; the outer
loop provides an unbiased generalization estimate. The random seed is fixed
(`RANDOM_SEED = 70`) throughout.

Global parameters (identical across experiments) are defined in `src/config.py`:
`RANDOM_SEED=70`, `N_BOOTSTRAP=500`, `N_PERMUT=1000`, `N_JOBS=-3`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point the code at your local data (see data/README.md)
export DATA_ROOT=/path/to/ADHD2026
export RESULTS_ROOT=./results
```

No absolute paths are hard-coded; every path is resolved through `src/config.py`
and can be overridden with environment variables.

## Repository layout

```
feature_extraction/         # MATLAB: Curvelet -> GGD feature extraction
src/
  config.py                 # all paths + global parameters
  experiments/              # main experiments (1–5) and leave-one-site-out
  sensitivity/              # confound / robustness analyses
  quality_control/          # motion (Euler number) and image-quality (PSNR/SSIM)
data/                       # not committed — see data/README.md
results/                    # script outputs (git-ignored)
figures/
```

## Pipeline

1. **Preprocessing** — FreeSurfer `recon-all` (v7.2.0) on all T1-weighted scans.
2. **Quality control** — visual inspection plus quantitative Euler number and
   PSNR/SSIM checks (`src/quality_control/`).
3. **Feature extraction** — multi-scale Curvelet transform (4 scales, 16
   orientations → 81 subbands) on 2D mosaics, each subband summarized by a
   Generalized Gaussian (α, β, μ) → 243-dim descriptor per region. Implemented
   in **MATLAB** under `feature_extraction/` (see that folder's README); the
   resulting feature tables are the input to the Python code in `src/`.
4. **Experiments and analyses** — Python scripts below.

## Feature extraction (`feature_extraction/`)

The MATLAB pipeline that turns FreeSurfer-preprocessed MRI into the
243-dimensional Curvelet/GGD feature vector per region. `characterization.m` is
the entry point; helper functions live alongside it. It requires CurveLab
(`fdct_wrapping`), the Tools for NIfTI/ANALYZE toolbox, and the
`ToolboxWaveletTexture` and `collage_utils` toolboxes. See
`feature_extraction/README.md` for requirements, inputs, and usage. The output
CSV (243 interleaved α, β, μ per subband) is the input expected by the Python
experiments.

## Table-to-script traceability

| Script | Produces (manuscript) |
|--------|------------------------|
| `feature_extraction/characterization.m` | Curvelet/GGD feature tables (input to all experiments) |
| `src/experiments/exp1_unsupervised.py` | Table 4 — unsupervised (k-means + SVM) |
| `src/experiments/exp2_supervised.py` | Table 5, Supp. S3 — supervised binary (nested CV) |
| `src/experiments/exp2_supervised_LOSO.py` | Supp. S9 — leave-one-site-out (Exp 2) |
| `src/experiments/exp3_cross_dataset.py` | Table 6 — cross-dataset (test on WMR-ADHD) |
| `src/experiments/exp3_cross_dataset_LOSO.py` | Supp. S10 — leave-one-site-out (Exp 3) |
| `src/experiments/exp4_intersite.py` | Table 7, Supp. S4, S7 — inter-site + Jaccard |
| `src/experiments/exp5_multiclass.py` | Tables 8–10, Supp. S5, S8 — multiclass subtypes |
| `src/sensitivity/age_stratified.py` | age-stratified (Children / Adolescents), Supp. S16 |
| `src/sensitivity/sex_stratified.py` | sex-stratified (Female / Male), Supp. S15 |
| `src/sensitivity/medication_naive.py` | medication-naive ADHD-C, Supp. S13 |
| `src/sensitivity/field_strength.py` | scanner field-strength (excl. NeuroIMAGE), Supp. S14 |
| `src/sensitivity/motion_matched.py` | motion-matched subsample, Supp. S11 |
| `src/quality_control/euler_adhd200.py` | FreeSurfer Euler number, ADHD-200 |
| `src/quality_control/euler_wmr.py` | FreeSurfer Euler number, WMR-ADHD |
| `src/quality_control/ssim_psnr.py` | PSNR/SSIM vs. MNI152 template |

Note: the leave-one-site-out analysis is implemented in the two dedicated
`*_LOSO.py` scripts above (Experiments 2 and 3), separate from the inter-site
classification in `exp4_intersite.py`.

## Data

See [`data/README.md`](data/README.md). No subject-level data is distributed;
both datasets are publicly available from their official sources.

## Citation

If you use this code, please cite the associated paper (Gonzalez, Tarquino &
Romero, Universidad Nacional de Colombia). Full reference will be added upon
publication.
