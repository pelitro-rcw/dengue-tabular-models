# AGENTS.md — guidance for an Antigravity agent in this repository

## Mission
Reproduce and (optionally) extend the sparse-data featurization pipeline:
two raw weekly columns (cases, rainfall) → ~105 latent features → TabPFN.

## How to run
```
pip install -r requirements.txt
py code/run.py --scale qc            # also: regional, country
```
If TabPFN login fails (Windows browser-auth bug), either set TABPFN_TOKEN (from
https://ux.priorlabs.ai, License tab) or run with the stand-in:
`$env:AUTOTABPFN_BACKEND="standin"` before the run command.

## Rules
1. Keep all feature engineering inside `code/enrich.py`. Features must be causal:
   at week t use only weeks <= t within the same unit-year; week-of-year
   baselines use prior years only. Never introduce future information.
2. Do not change the evaluation protocol in `code/evaluate.py` (leave-one-year-out,
   train-only fitting). It is what keeps the numbers honest.
3. Never report a number you did not produce. Quote `outputs/<scale>/metrics.json`
   and state the `backend`. Stand-in numbers are not TabPFN numbers.
4. The story is the *lift* from raw 2 columns to the enriched set, not the
   absolute score. Always report both rows.

## Good extension tasks
- Add a new mechanism-grounded feature family to `enrich.py` and report whether
  the enriched AUROC/AUPRC improves.
- Merge an external ONI (ENSO) series and add rainfall × ONI features.
- Add per-region calibration or a leave-one-unit-out variant.

## Statistics layer (Delphi revision) — rules
5. Model-vs-model inference is PAIRED + CLUSTERED: use `stats.paired_cluster_bootstrap`
   with resampling unit = **unit-year** (`cluster` in `oof_all.npz`). Never resample
   weeks; never use an unpaired/row-level test for these comparisons.
6. Keep OOF saving for EVERY representation (`oof_all.npz`); paired tests and per-region
   CIs depend on it.
7. Exactly one confirmatory contrast (enriched − raw, AUROC). Everything else —
   including the rain-threshold dynamics−accumulation lift — is exploratory and FDR/Holm
   controlled.
8. Report effective N (year-clusters) beside every CI; wide CIs at qc are correct.
9. The comparator and the rain dynamics-vs-accumulation lift must be compared on the
   COMMON weeks only (`n_common`), not mismatched samples.
10. Stand-in stats are not TabPFN stats. Shipped `outputs/` are standin demos;
    `outputs_published_tabpfn/` holds the original TabPFN headline.
