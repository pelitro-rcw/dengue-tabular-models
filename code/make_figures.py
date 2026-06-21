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
from sklearn.feature_selection import mutual_info_classif
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


def fig1(scales):
    n = len(scales)
    fig, ax = plt.subplots(n, 2, figsize=(11, 3.0 * n), squeeze=False)
    for i, s in enumerate(scales):
        M = _metrics(s)
        # (a) representation >> model: TabPFN raw vs TS-FM raw cases vs TabPFN enriched
        tfm = M.get("tfm_comparator", {})
        comp_name = tfm.get("name", "ts model")
        names = ["TabPFN\nraw 2 cols", f"{comp_name}\nraw cases", "TabPFN\n+ enriched"]
        au = [M["raw_2col"]["auroc"], tfm.get("auroc", float("nan")), M["enriched"]["auroc"]]
        cols = ["#c2c2c2", "#7fb3d5", "#d95f02"]
        b = ax[i][0].bar(names, au, color=cols)
        for bar in b:
            h = bar.get_height()
            if np.isfinite(h):
                ax[i][0].text(bar.get_x() + bar.get_width()/2, h+0.005,
                              f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        ax[i][0].axhline(0.5, ls=":", color="grey")
        ax[i][0].set_ylim(0.4, 1.0); ax[i][0].set_ylabel(f"{SCALE_LABEL[s]}\nAUROC")
        ax[i][0].tick_params(axis="x", labelsize=7)
        if i == 0:
            ax[i][0].set_title("a  Representation \u226b model")
        # (b) enrichment ladder
        lad = M.get("ladder", [])
        if lad:
            x = range(len(lad))
            ax[i][1].plot(x, [r["auroc"] for r in lad], "-o", color="#d95f02", label="AUROC")
            ax[i][1].plot(x, [r["auprc"] for r in lad], "--s", color="#1b9e77", label="AUPRC")
            ax[i][1].set_xticks(list(x))
            ax[i][1].set_xticklabels([r["rung"].replace("+ ", "+\n") for r in lad], fontsize=7)
            ax[i][1].set_ylim(0.4, 0.95); ax[i][1].grid(alpha=.25)
            if i == 0:
                ax[i][1].set_title("b  Domain-informed enrichment ladder"); ax[i][1].legend(fontsize=8)
    fig.suptitle(f"Fig 1 — representation \u226b model & enrichment ladder (backend: {_metrics(scales[0])['backend']})")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, "fig1_representation_vs_model.png"), dpi=160)
    print("  fig1_representation_vs_model.png")


def fig2(scales):
    n = len(scales)
    fig, ax = plt.subplots(n, 1, figsize=(8.5, 4.2 * n), squeeze=False)
    for i, s in enumerate(scales):
        enr = pd.read_csv(os.path.join(OUT, s, "enriched.csv"))
        sub = enr[enr.get("labelable", 1) == 1] if "labelable" in enr else enr
        feats = [c for c in sub.columns if c not in META and sub[c].dtype != object]
        X = SimpleImputer(strategy="median").fit_transform(sub[feats])
        y = sub["y"].to_numpy()
        mi = mutual_info_classif(X, y, random_state=0)
        order = np.argsort(mi)[::-1][:14]
        names = [feats[j] for j in order][::-1]; vals = mi[order][::-1]
        colors = [FAM_COLORS.get(family_of(nm), "#999999") for nm in names]
        ax[i][0].barh(names, vals, color=colors)
        ax[i][0].set_xlabel("mutual information with pre-peak label")
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
    rm = rm.sort_values(["climate_type", "auroc"], ascending=[True, False])
    colors = {"I": "#1f6fb0", "II": "#3fb0b0", "III": "#a8d5a0", "IV": "#f5a623"}
    means = rm.groupby("climate_type", observed=True)["auroc"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(rm)), rm["auroc"], color=[colors[t] for t in rm["climate_type"]])
    ax.set_xticks(range(len(rm))); ax.set_xticklabels(rm["REGION"], rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, ls=":", color="grey"); ax.set_ylabel("AUROC (LOYO)"); ax.set_ylim(0.35, 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in order]
    ax.legend(handles, [f"Type {t}" for t in order], loc="lower left", ncol=4, fontsize=8)
    title = "  ".join(f"Type {t}: {means.get(t, float('nan')):.2f}" for t in order)
    ax.set_title(f"Fig 3 — predictability tracks climate regime   ({title})")
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
        if i == 0:
            ax[i][0].set_title(f"a  Calibration (ECE={_ece(y, p):.3f})")
        else:
            ax[i][0].set_title(f"ECE={_ece(y, p):.3f}", fontsize=9)
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
