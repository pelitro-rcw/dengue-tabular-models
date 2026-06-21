"""
run.py — reproduce the sparse-data -> featurization -> TabPFN result and write
the COMPLETE output set (CSVs + predictions; figures via make_figures.py).

Usage:
  py code/run.py --scale qc
  py code/run.py --scale regional
  AUTOTABPFN_BACKEND=standin py code/run.py --scale qc   # no account needed

Per scale, writes outputs/<scale>/:
  metrics.json          raw 2-col vs case-only vs enriched (the enrichment ladder)
  enriched.csv          the ~105-feature latent reconstruction
  feature_dictionary.csv every feature + source + one-line rationale
  preds_enriched.npz     out-of-fold y_true / y_prob (for the calibration figure)
For regional/country also:
  regional_metrics.csv   per-unit LOYO metrics + PAGASA climate type
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")            # quiet TabPFN/sklearn notices; outputs unchanged
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from labels import build_peak_labels
from enrich import (enrich, feature_columns, feature_dictionary,
                    case_calendar_columns, ladder_groups, RAW_TWO)
from evaluate import loyo, loyo_per_region, get_backend

SCALES = {
    "qc":       dict(sheet="QC Data",       case="DC_QC",          rain="RF_NASA", groups=[]),
    "regional": dict(sheet="Regional Data", case="DC_DOH",         rain="RF_HDX",  groups=["REGION"]),
    "country":  dict(sheet="Country Data",  case="DC_OPENDENGUE",  rain="RF_NASA", groups=["COUNTRY"]),
}

# Best-effort DOMINANT PAGASA climate type per administrative region.
# PAGASA types are sub-regional; edit outputs/region_climate_type.csv to refine.
CLIMATE_TYPE = {
    "REGION I": "I", "NCR": "I", "CAR": "I", "REGION III": "I",
    "REGION IV-A": "I", "MIMAROPA": "I", "REGION VI": "I",
    "REGION II": "III", "REGION VII": "III", "REGION IX": "III",
    "REGION V": "II", "REGION VIII": "II", "REGION XIII": "II",
    "REGION X": "IV", "REGION XI": "IV", "REGION XII": "IV", "BARMM": "IV",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="qc")
    ap.add_argument("--xlsx", default=os.path.join(ROOT, "data", "Dengue-Rainfall_Dataset.xlsx"))
    ap.add_argument("--fast", action="store_true",
                    help="cheaper compute (1 estimator, no calibration, smaller context); "
                         "SAME complete outputs, just faster")
    args = ap.parse_args()

    if args.fast:
        os.environ["N_EST"] = os.environ.get("N_EST", "1")
        os.environ["TABPFN_MAX_TRAIN"] = os.environ.get("TABPFN_MAX_TRAIN", "500")
        os.environ["AUTOTABPFN_CALIBRATE"] = "0"
        print("[fast mode] 1 estimator, no calibration, max_train=500 — full outputs kept")

    cfg = SCALES[args.scale]; groups = cfg["groups"]

    df = pd.read_excel(args.xlsx, sheet_name=cfg["sheet"])
    for col in ("FLAG_COVID", "FLAG_TERMINAL_GAP"):
        if col in df.columns:
            frac = df.groupby("YR")[col].mean()
            df = df[~df["YR"].isin(frac[frac >= 0.5].index)].copy()
    if "FLAG_PLAUSIBILITY" in df.columns:
        df = df[df["FLAG_PLAUSIBILITY"] == 0].copy()

    print(f"[{args.scale}] building ~105 features from 2 raw columns...")
    enr = enrich(df, cfg["case"], cfg["rain"], groups)
    from catch22_features import add_catch22
    enr, c22_method, c22_cols = add_catch22(enr, cfg["case"], cfg["rain"], groups)
    lab = build_peak_labels(df, cfg["case"], groups)
    enr = enr.merge(lab[groups + ["YR", "WN", "y", "labelable"]], on=groups + ["YR", "WN"])

    cols_all = feature_columns(enr, cfg["case"], cfg["rain"], groups)      # domain + catch22
    cols_domain = [c for c in cols_all if not c.startswith(("c22_", "c22fb_"))]
    cols_caseonly = case_calendar_columns(cols_domain)
    cols_c22only = RAW_TWO + c22_cols
    backend, fp = get_backend()
    print(f"backend = {backend} | features = {len(cols_all)} "
          f"(domain {len(cols_domain)} + catch22 {len(c22_cols)} via {c22_method})")

    outdir = os.path.join(ROOT, "outputs", args.scale)
    os.makedirs(outdir, exist_ok=True)

    # representations: raw -> case-only -> catch22-only -> domain -> domain+catch22(full)
    res_raw = loyo(enr, RAW_TWO, fp)
    res_case = loyo(enr, cols_caseonly, fp)
    res_c22 = loyo(enr, cols_c22only, fp)
    res_dom = loyo(enr, cols_domain, fp)
    res_full, Y, P = loyo(enr, cols_all, fp, return_oof=True)
    res_enr = res_full

    # enrichment ladder (fig1b): raw -> +case -> +climate -> +transmission biology -> +catch22
    grp = ladder_groups(cols_domain)
    r1 = RAW_TWO
    r2 = RAW_TWO + grp["case_view"]
    r3 = r2 + grp["climate"]
    r4 = cols_domain
    r5 = cols_all
    rungs = [("raw 2 cols", r1), ("+ case views", r2),
             ("+ climate", r3), ("+ transmission biology", r4), ("+ catch22", r5)]
    ladder = []
    for name, cs in rungs:
        rr = loyo(enr, cs, fp)
        ladder.append(dict(rung=name, n=rr["n_features"], auroc=rr["auroc"], auprc=rr["auprc"]))

    out = dict(scale=args.scale, backend=backend, catch22_method=c22_method,
               raw_2col=res_raw, case_only=res_case, catch22_only=res_c22,
               domain_only=res_dom, enriched=res_enr, ladder=ladder,
               lift_auroc=res_enr["auroc"] - res_raw["auroc"],
               lift_auprc=res_enr["auprc"] - res_raw["auprc"])

    # time-series FM comparator (fig1a third bar): forecast cases -> pre-peak score
    from tfm_compare import tfm_score
    tfm = tfm_score(enr, cfg["case"], groups)
    out["tfm_comparator"] = tfm
    print(f"  comparator: {tfm['name']} raw-cases AUROC {tfm['auroc']:.3f}")

    # CSV + prediction outputs
    enr.to_csv(os.path.join(outdir, "enriched.csv"), index=False)
    feature_dictionary(cols_all).to_csv(os.path.join(outdir, "feature_dictionary.csv"), index=False)
    np.savez(os.path.join(outdir, "preds_enriched.npz"), y_true=Y, y_prob=P)
    json.dump(out, open(os.path.join(outdir, "metrics.json"), "w"), indent=2, default=float)

    # per-region metrics + climate type (multi-unit scales) — kept for full output
    if groups:
        gc = groups[0]
        cm = CLIMATE_TYPE if gc == "REGION" else None
        per = loyo_per_region(enr, cols_all, gc, fp, climate_map=cm)
        per.to_csv(os.path.join(outdir, "regional_metrics.csv"), index=False)
        if cm is not None:
            pd.DataFrame(sorted(cm.items()), columns=["REGION", "climate_type"]).to_csv(
                os.path.join(ROOT, "outputs", "region_climate_type.csv"), index=False)
        print(f"  wrote regional_metrics.csv ({len(per)} units)")

    print("\n  setting            feats   AUROC   AUPRC   Brier")
    print("  " + "-" * 48)
    for tag, r in (("raw 2 columns", res_raw), ("case-only", res_case),
                   ("catch22-only", res_c22), ("domain", res_dom),
                   ("domain+catch22", res_full)):
        print(f"  {tag:16s}  {r['n_features']:4d}   {r['auroc']:.3f}   {r['auprc']:.3f}   {r['brier']:.3f}")
    print(f"\n  catch22 backend: {c22_method}")
    print(f"  lift (raw->full):  +{out['lift_auroc']:.3f} AUROC  +{out['lift_auprc']:.3f} AUPRC")
    print(f"  backend={backend}; outputs in outputs/{args.scale}/")
    print("  next: py code/make_figures.py")


if __name__ == "__main__":
    main()
