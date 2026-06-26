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
  rain_thresholds.csv    rainfall trigger preceding the TA onset. Headline = interpretable
                         accumulation threshold in mm (rain >= T, Youden + sens/spec/AUROC,
                         LOYO). Secondary dyn_* columns = best rainfall-dynamics
                         representation (rate / STA-LTA) documenting the AUROC lift
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
    ap.add_argument("--stability", type=int, default=1,
                    help="K>1 re-runs raw+enriched across K seeds to report Monte-Carlo "
                         "SD of AUROC/AUPRC (algorithmic variance, distinct from the data bootstrap)")
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
    # Delphi revision: keep out-of-fold predictions for EVERY representation (not just
    # enriched) so model-vs-model comparisons can be done PAIRED.
    reps = {"raw_2col": RAW_TWO, "case_only": cols_caseonly, "catch22_only": cols_c22only,
            "domain_only": cols_domain, "enriched": cols_all}
    res_by, oof_P = {}, {}
    Y = C = W = None
    for nm, cs in reps.items():
        r, Yi, Pi, Ci, Wi = loyo(enr, cs, fp, return_oof=True, group_cols=groups)
        res_by[nm] = r; oof_P[nm] = Pi
        if Y is None:
            Y, C, W = Yi, Ci, Wi
    res_raw, res_case = res_by["raw_2col"], res_by["case_only"]
    res_c22, res_dom = res_by["catch22_only"], res_by["domain_only"]
    res_full = res_enr = res_by["enriched"]
    P = oof_P["enriched"]

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
    tfm, tfm_rows = tfm_score(enr, cfg["case"], groups, return_rows=True)
    out["tfm_comparator"] = tfm
    print(f"  comparator: {tfm['name']} raw-cases AUROC {tfm['auroc']:.3f}")

    # CSV + prediction outputs
    enr.to_csv(os.path.join(outdir, "enriched.csv"), index=False)
    feature_dictionary(cols_all).to_csv(os.path.join(outdir, "feature_dictionary.csv"), index=False)
    np.savez(os.path.join(outdir, "preds_enriched.npz"), y_true=Y, y_prob=P)
    # Delphi revision: persist OOF for ALL representations + cluster id (unit-year) + week.
    np.savez(os.path.join(outdir, "oof_all.npz"),
             y_true=Y, cluster=C.astype(str), wn=W,
             **{f"p_{nm}": oof_P[nm] for nm in reps})
    json.dump(out, open(os.path.join(outdir, "metrics.json"), "w"), indent=2, default=float)

    # ---------------------------------------------------------------- statistics layer
    B = int(os.environ.get("BOOT_B", "2000"))
    import stats as st
    statout = dict(scale=args.scale, backend=backend, B=B,
                   resampling_unit="unit-year", n_rows=int(len(Y)),
                   n_clusters=int(len(np.unique(C))))

    # (A) paired representation comparison
    rep_order = list(reps)
    pcb = st.paired_cluster_bootstrap(Y, {k: oof_P[k] for k in rep_order}, C, B=B)
    pairs = [(a, b) for i, a in enumerate(rep_order) for b in rep_order[i + 1:]]
    delong = {}
    for a, b in pairs:
        _, _, z, p = st.delong_roc_test(Y, oof_P[a], oof_P[b])
        delong[f"{a}_vs_{b}"] = dict(z=z, p=p)
    conf_key = "raw_2col_vs_enriched" if "raw_2col_vs_enriched" in pcb["auroc"]["delta"] else "enriched_vs_raw_2col"
    expl = [k for k in pcb["auroc"]["delta"] if k != conf_key]
    expl_p = [pcb["auroc"]["delta"][k]["p"] for k in expl]
    fdr = st.bh_fdr(expl_p) if expl_p else []
    statout["representation"] = dict(
        bootstrap=pcb, delong=delong,
        confirmatory=dict(contrast=conf_key, metric="auroc", **pcb["auroc"]["delta"][conf_key]),
        exploratory_fdr={k: float(q) for k, q in zip(expl, fdr)})

    # (C) comparator PAIRED on common (cluster, wn) weeks vs enriched
    cls = {(c, int(w)): i for i, (c, w) in enumerate(zip(C.astype(str), W))}
    common = [(cls[(c, int(w))], j) for j, (c, w) in
              enumerate(zip(tfm_rows["cluster"].astype(str), tfm_rows["wn"])) if (c, int(w)) in cls]
    if len(common) >= 30:
        ci_idx = np.array([i for i, _ in common]); tj = np.array([j for _, j in common])
        cmp_pcb = st.paired_cluster_bootstrap(
            Y[ci_idx], {"enriched": P[ci_idx], tfm["name"]: tfm_rows["score"][tj]},
            C.astype(str)[ci_idx], B=B, metric_names=["auroc", "auprc"])
        statout["comparator_paired"] = dict(
            n_common=int(len(common)), aligned_on="(unit-year, week)",
            note="enriched vs comparator on weeks BOTH can score; the comparator's raw "
                 "bar in metrics.json is on a different sample.", bootstrap=cmp_pcb)
    else:
        statout["comparator_paired"] = dict(n_common=int(len(common)), note="too few common weeks")

    # (E) calibration of the enriched model
    statout["calibration_enriched"] = st.calibration_block(Y, P, C, B=B)

    # rainfall trigger level that precedes the onset of transmission acceleration
    try:
        from rain_threshold import derive as _derive_thr
        _thr_df, _thr_stats = _derive_thr(args.scale, enr, boot_B=B)
        if _thr_stats:
            statout["rain_threshold"] = _thr_stats
    except Exception as e:
        print(f"  rain_threshold skipped: {e}")

    # per-region metrics + climate type (multi-unit scales) — kept for full output
    if groups:
        gc = groups[0]
        cm = CLIMATE_TYPE if gc == "REGION" else None
        per, per_oof = loyo_per_region(enr, cols_all, gc, fp, climate_map=cm, return_oof=True)
        # Delphi revision: per-region AUROC CI (cluster bootstrap over years) from the
        # SAME region-only predictions as the bar, + a vs-0.5 chance flag.
        ci_lo, ci_hi, beats, ncl = [], [], [], []
        for unit in per[gc].astype(str):
            yo = per_oof.get(unit)
            if yo is not None and len(yo[0]) >= 5 and len(np.unique(yo[0])) > 1:
                r = st.auroc_ci_vs_chance(yo[0], yo[1], yo[2], B=B)
                ci_lo.append(r["ci"][0]); ci_hi.append(r["ci"][1])
                beats.append(r["beats_chance"]); ncl.append(r["n_year_clusters"])
            else:
                ci_lo.append(np.nan); ci_hi.append(np.nan); beats.append(False); ncl.append(0)
        per["auroc_ci_lo"] = ci_lo; per["auroc_ci_hi"] = ci_hi
        per["beats_chance"] = beats; per["n_year_clusters"] = ncl
        per.to_csv(os.path.join(outdir, "regional_metrics.csv"), index=False)
        if "climate_type" in per:
            statout["climate_gradient"] = st.kruskal_gradient(
                per["auroc"].to_numpy(), per["climate_type"].astype(str).to_numpy())
        if cm is not None:
            pd.DataFrame(sorted(cm.items()), columns=["REGION", "climate_type"]).to_csv(
                os.path.join(ROOT, "outputs", "region_climate_type.csv"), index=False)
        nbc = int(np.sum(~np.array(beats)))
        print(f"  wrote regional_metrics.csv ({len(per)} units; {nbc} do NOT clear 0.5 by CI)")

    # (F) stability: Monte-Carlo SD across seeds (algorithmic variance)
    if args.stability and args.stability > 1:
        base_seed = os.environ.get("PIPELINE_SEED")
        seed_rows = []
        for sd in range(args.stability):
            os.environ["PIPELINE_SEED"] = str(sd)
            r_raw = loyo(enr, RAW_TWO, fp); r_enr = loyo(enr, cols_all, fp)
            seed_rows.append(dict(seed=sd, raw_auroc=r_raw["auroc"], enr_auroc=r_enr["auroc"],
                                  lift=r_enr["auroc"] - r_raw["auroc"]))
        if base_seed is None:
            os.environ.pop("PIPELINE_SEED", None)
        else:
            os.environ["PIPELINE_SEED"] = base_seed
        lifts = np.array([r["lift"] for r in seed_rows])
        statout["stability"] = dict(K=args.stability, seeds=seed_rows,
            enr_auroc_sd=float(np.std([r["enr_auroc"] for r in seed_rows], ddof=1)),
            lift_mean=float(lifts.mean()), lift_sd=float(lifts.std(ddof=1)))
        print(f"  [stability] K={args.stability}  lift {lifts.mean():+.3f} ± {lifts.std(ddof=1):.3f} (seed SD)")

    json.dump(statout, open(os.path.join(outdir, "stats.json"), "w"), indent=2, default=float)
    d = statout["representation"]["confirmatory"]
    print(f"  [stats] confirmatory {conf_key} ΔAUROC={d['delta']:+.3f} "
          f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p']:.4f} "
          f"(clusters={statout['n_clusters']}, B={B})")

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
