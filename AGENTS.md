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
