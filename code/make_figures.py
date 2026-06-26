"""
make_figures.py — multi-panel figures, one ROW per scale (QC / regional / country).

  fig1  3 rows x 2 cols : (a) representation comparison bars, (b) enrichment ladder
  fig2  3 rows x 1 col  : feature attribution (mutual information), coloured by family
  fig3  regional only   : per-region AUROC by PAGASA climate type
  fig4  3 rows x 2 cols : (a) calibration reliability, (b) score separation

Run after the scales:
  py code/run.py --scale qc --fast
  py code/run.py --scale regional --fast
  py code/run.py --scale country --fast
  py code/make_figures.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")            # quiet sklearn/joblib parallel notices; outputs unchanged
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score as _auc_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE); OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
from enrich import family_of  # noqa

SCALE_ORDER = ["qc", "regional", "country"]
SCALE_LABEL = {"qc": "Quezon City", "regional": "Regional", "country": "Country"}
META = {"YR", "WN", "y", "labelable", "REGION", "COUNTRY",
        "DC_QC", "DC_DOH", "DC_OPENDENGUE", "RF_NASA", "RF_HDX"}
FAM_COLORS = {"case": "#1b9e77", "climate": "#7570b3", "season": "#666666",
              "transmission": "#d95f02", "early-warning": "#e7298a", "catch22": "#1f78b4"}


def present_scales():
    return [s for s in SCALE_ORDER if os.path.exists(os.path.join(OUT, s, "metrics.json"))]


def _metrics(s):
    return json.load(open(os.path.join(OUT, s, "metrics.json")))


def _stats(s):
    p = os.path.join(OUT, s, "stats.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _star(p):
    if p is None or not np.isfinite(p):
        return "n/s"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def fig1(scales):
    n = len(scales)
    fig, ax = plt.subplots(n, 2, figsize=(11, 3.0 * n), squeeze=False)
    for i, s in enumerate(scales):
        M = _metrics(s); S = _stats(s)
        tfm = M.get("tfm_comparator", {})
        comp_name = tfm.get("name", "ts model")
        names = ["TabPFN\nraw 2 cols", f"{comp_name}\nraw cases", "TabPFN\ncatch22", "TabPFN\n+ enriched"]
        au = [M["raw_2col"]["auroc"], tfm.get("auroc", float("nan")),
              M["catch22_only"]["auroc"], M["enriched"]["auroc"]]
        cols = ["#c2c2c2", "#7fb3d5", "#1f78b4", "#d95f02"]
        # bootstrap CIs for the three TabPFN bars; comparator CI on common weeks
        yerr = np.full((2, 4), np.nan)
        if S:
            rci = S["representation"]["bootstrap"]["auroc"]["ci"]
            for col, key in ((0, "raw_2col"), (2, "catch22_only"), (3, "enriched")):
                lo, hi = rci[key]; yerr[0, col] = au[col] - lo; yerr[1, col] = hi - au[col]
            cmp = S.get("comparator_paired", {})
            if "bootstrap" in cmp and comp_name in cmp["bootstrap"]["auroc"]["ci"]:
                lo, hi = cmp["bootstrap"]["auroc"]["ci"][comp_name]
                yerr[0, 1] = max(0, au[1] - lo); yerr[1, 1] = max(0, hi - au[1])
        b = ax[i][0].bar(names, au, color=cols, yerr=yerr, capsize=4,
                         error_kw=dict(ecolor="#333", lw=1))
        for bar in b:
            h = bar.get_height()
            if np.isfinite(h):
                ax[i][0].text(bar.get_x() + bar.get_width()/2, h+0.005,
                              f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        if S:
            conf = S["representation"]["confirmatory"]
            yb = max(au[0], au[3]) + 0.12
            ax[i][0].plot([0, 0, 3, 3], [yb-0.02, yb, yb, yb-0.02], lw=1, color="#333")
            ax[i][0].text(1.5, yb+0.005,
                          f"raw vs enriched: Δ={abs(conf['delta']):.3f} {_star(conf['p'])} (p={conf['p']:.3f})",
                          ha="center", va="bottom", fontsize=7.5)
        ax[i][0].axhline(0.5, ls=":", color="grey")
        ax[i][0].set_ylim(0.4, 1.18); ax[i][0].set_ylabel(f"{SCALE_LABEL[s]}\nAUROC")
        ax[i][0].tick_params(axis="x", labelsize=7)
        if i == 0:
            ax[i][0].set_title("a  Representation \u226b model  (95% cluster-bootstrap CI)")
        if S and "comparator_paired" in S:
            ax[i][0].text(0.5, 0.02, f"comparator paired on n={S['comparator_paired'].get('n_common')} "
                          f"common weeks; its raw bar (metrics.json) is on a different sample",
                          transform=ax[i][0].transAxes, fontsize=6, color="#666", ha="center")
        lad = M.get("ladder", [])
        if lad:
            x = range(len(lad)); aur = [r["auroc"] for r in lad]
            ax[i][1].plot(x, aur, "-o", color="#d95f02", label="AUROC")
            ax[i][1].plot(x, [r["auprc"] for r in lad], "--s", color="#1b9e77", label="AUPRC")
            for k in range(1, len(aur)):
                if aur[k] < aur[k-1] - 1e-9:
                    ax[i][1].annotate("dip", (k, aur[k]), textcoords="offset points",
                                      xytext=(0, -12), fontsize=6.5, color="#b00", ha="center")
            ax[i][1].set_xticks(list(x))
            ax[i][1].set_xticklabels([r["rung"].replace("+ ", "+\n") for r in lad], fontsize=7)
            ax[i][1].set_ylim(0.4, 0.95); ax[i][1].grid(alpha=.25)
            if i == 0:
                ax[i][1].set_title("b  Enrichment ladder (rung Δ exploratory, FDR-controlled)")
                ax[i][1].legend(fontsize=8)
    fig.suptitle(f"Fig 1 — representation \u226b model & enrichment ladder (backend: {_metrics(scales[0])['backend']})")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, "fig1_representation_vs_model.png"), dpi=160)
    print("  fig1_representation_vs_model.png")


def _predictive_importance(X, y, groups):
    """Leave-one-year-out permutation importance (delta AUROC) — predictive
    contribution, not marginal association. Falls back to univariate AUROC."""
    try:
        imp = np.zeros(X.shape[1]); nsplit = min(4, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=max(2, nsplit))
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[te])) < 2:
                continue
            rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1).fit(X[tr], y[tr])
            r = permutation_importance(rf, X[te], y[te], n_repeats=4, random_state=0, scoring="roc_auc", n_jobs=1)
            imp += r.importances_mean
        if np.allclose(imp, 0):
            raise ValueError("degenerate")
        return imp, "leave-one-year-out permutation importance (Δ AUROC)"
    except Exception:
        imp = np.array([abs(_auc_score(y, X[:, j]) - 0.5) for j in range(X.shape[1])])
        return imp, "univariate predictive AUROC (|AUROC − 0.5|)"


def fig2(scales):
    n = len(scales)
    fig, ax = plt.subplots(n, 1, figsize=(8.5, 4.2 * n), squeeze=False)
    xlabel = "predictive importance"
    for i, s in enumerate(scales):
        enr = pd.read_csv(os.path.join(OUT, s, "enriched.csv"))
        sub = enr[enr.get("labelable", 1) == 1] if "labelable" in enr else enr
        feats = [c for c in sub.columns if c not in META and sub[c].dtype != object]
        X = SimpleImputer(strategy="median").fit_transform(sub[feats])
        y = sub["y"].to_numpy()
        groups = sub["YR"].to_numpy()
        imp, xlabel = _predictive_importance(X, y, groups)
        order = np.argsort(imp)[::-1][:14]
        names = [feats[j] for j in order][::-1]; vals = imp[order][::-1]
        colors = [FAM_COLORS.get(family_of(nm), "#999999") for nm in names]
        ax[i][0].barh(names, vals, color=colors)
        ax[i][0].set_xlabel(xlabel)
        ax[i][0].set_title(f"{SCALE_LABEL[s]} — what carries the early-warning signal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLORS.values()]
    fig.legend(handles, list(FAM_COLORS), loc="lower center", ncol=len(FAM_COLORS), fontsize=8)
    fig.suptitle("Fig 2 — feature attribution by scale")
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.97]); fig.savefig(os.path.join(FIG, "fig2_feature_attribution.png"), dpi=160)
    print("  fig2_feature_attribution.png")


def fig3():
    p = os.path.join(OUT, "regional", "regional_metrics.csv")
    if not os.path.exists(p):
        print("  (fig3 skipped — run: py code/run.py --scale regional)"); return
    rm = pd.read_csv(p)
    if "climate_type" not in rm:
        return
    order = ["I", "II", "III", "IV"]
    rm["climate_type"] = pd.Categorical(rm["climate_type"], order, ordered=True)
    rm = rm.sort_values(["climate_type", "auroc"], ascending=[True, False]).reset_index(drop=True)
    colors = {"I": "#1f6fb0", "II": "#3fb0b0", "III": "#a8d5a0", "IV": "#f5a623"}
    means = rm.groupby("climate_type", observed=True)["auroc"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(rm)); yerr = None
    if {"auroc_ci_lo", "auroc_ci_hi"}.issubset(rm.columns):
        lo = (rm["auroc"] - rm["auroc_ci_lo"]).clip(lower=0)
        hi = (rm["auroc_ci_hi"] - rm["auroc"]).clip(lower=0)
        yerr = np.vstack([lo.to_numpy(), hi.to_numpy()])
    bars = ax.bar(x, rm["auroc"], color=[colors[t] for t in rm["climate_type"]],
                  yerr=yerr, capsize=3, error_kw=dict(ecolor="#333", lw=0.8))
    if "beats_chance" in rm.columns:
        for j, ok in enumerate(rm["beats_chance"]):
            if not ok:
                bars[j].set_hatch("///"); bars[j].set_edgecolor("#b00")
    ax.set_xticks(list(x)); ax.set_xticklabels(rm["REGION"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, ls=":", color="grey"); ax.set_ylabel("AUROC (LOYO)"); ax.set_ylim(0.35, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in order]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="#b00", hatch="///"))
    ax.legend(handles, [f"Type {t}" for t in order] + ["CI incl. 0.5"], loc="lower left", ncol=5, fontsize=8)
    title = "  ".join(f"Type {t}: {means.get(t, float('nan')):.2f}" for t in order)
    S = _stats("regional"); grad = (S or {}).get("climate_gradient", {})
    gtxt = ""
    if grad and np.isfinite(grad.get("p", np.nan)):
        gtxt = f"\nKruskal–Wallis across climate types: H={grad['H']:.2f}, p={grad['p']:.3f} (exploratory; climate type is best-effort dominant)"
    ax.set_title(f"Fig 3 — predictability vs climate regime   ({title}){gtxt}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_regional_climate_gradient.png"), dpi=160)
    print("  fig3_regional_climate_gradient.png")


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def fig4(scales):
    scales = [s for s in scales if os.path.exists(os.path.join(OUT, s, "preds_enriched.npz"))]
    n = len(scales)
    fig, ax = plt.subplots(n, 2, figsize=(10, 3.2 * n), squeeze=False)
    for i, s in enumerate(scales):
        d = np.load(os.path.join(OUT, s, "preds_enriched.npz"))
        y, p = d["y_true"], d["y_prob"]
        nb = min(10, max(3, len(np.unique(p))))
        frac, mean = calibration_curve(y, p, n_bins=nb, strategy="quantile")
        ax[i][0].plot([0, 1], [0, 1], "--", color="grey")
        ax[i][0].plot(mean, frac, "-o", color="#d95f02")
        ax[i][0].set_xlabel("predicted P(pre-peak)"); ax[i][0].set_ylabel(f"{SCALE_LABEL[s]}\nobserved freq")
        S = _stats(s); cal = (S or {}).get("calibration_enriched", {})
        if cal:
            sub = (f"ECE={cal['ece']:.3f} [{cal['ece_ci'][0]:.3f},{cal['ece_ci'][1]:.3f}]  "
                   f"slope={cal['cox_slope']:.2f} [{cal['cox_slope_ci'][0]:.2f},{cal['cox_slope_ci'][1]:.2f}]  "
                   f"Spiegelhalter p={cal['spiegelhalter_p']:.3f}")
        else:
            sub = f"ECE={_ece(y, p):.3f}"
        ax[i][0].set_title(("a  Calibration (95% cluster-bootstrap CI)\n" if i == 0 else "") + sub,
                           fontsize=8.5 if i == 0 else 7.5)
        ax[i][1].hist(p[y == 0], bins=20, density=True, alpha=.6, color="#c2c2c2", label="non pre-peak")
        ax[i][1].hist(p[y == 1], bins=20, density=True, alpha=.6, color="#d95f02", label="pre-peak")
        ax[i][1].set_xlabel("predicted probability"); ax[i][1].set_ylabel("density")
        if i == 0:
            ax[i][1].set_title("b  Score separation"); ax[i][1].legend(fontsize=8)
    fig.suptitle("Fig 4 — calibration & score separation by scale")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, "fig4_calibration.png"), dpi=160)
    print("  fig4_calibration.png")


def main():
    scales = present_scales()
    if not scales:
        print("No metrics found. Run: py code/run.py --scale qc --fast"); return
    print("Figures ->", FIG, "| scales:", scales)
    fig1(scales)
    fig2(scales)
    fig3()
    fig4(scales)


if __name__ == "__main__":
    main()
