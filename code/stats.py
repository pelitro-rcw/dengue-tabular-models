"""
stats.py — inference layer added in the Delphi revision.

Everything here treats the LOYO out-of-fold predictions as PAIRED (same test
weeks score every representation) and CLUSTERED (weeks nested in unit-years).
The workhorse is a paired *cluster* bootstrap whose resampling unit is the
unit-year, never the individual week. Closed-form companions (DeLong for AUROC,
Spiegelhalter for calibration) are provided as secondary cross-checks.

Pure numpy/scipy/sklearn — no network, no extra installs.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------------- metrics
def _auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def _auprc(y, p):
    return average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan

def _brier(y, p):
    return brier_score_loss(y, p)

METRICS = {"auroc": _auroc, "auprc": _auprc, "brier": _brier}
# brier is a loss (lower better); for Δ direction we flip its sign downstream.
LOWER_BETTER = {"brier"}


# ----------------------------------------------------------------------------- cluster bootstrap
def _cluster_index(clusters, rng):
    """Sample whole clusters with replacement; return concatenated row indices."""
    uniq = np.unique(clusters)
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    # precompute member rows per cluster once per call is cheap enough at our scale
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    return np.concatenate([members[c] for c in picked])


def paired_cluster_bootstrap(y, preds: dict, clusters, B=2000, seed=0, metric_names=None):
    """
    y          : (n,) int labels
    preds      : {name: (n,) prob}  — all share the same rows / clusters
    clusters   : (n,) cluster id (unit-year) — the resampling unit
    metric_names : subset of METRICS (default all). Use rank-only (auroc/auprc) when
                   a "model" outputs scores rather than calibrated probabilities.
    Returns dict with, per metric:
      point     : {name: value}
      ci        : {name: [lo, hi]}      (2.5/97.5 percentile)
      delta     : {f"{a}_vs_{b}": {"delta":, "ci":[lo,hi], "p":}}
    p is a two-sided cluster-bootstrap p for H0: Δmetric = 0.
    """
    y = np.asarray(y); clusters = np.asarray(clusters)
    names = list(preds)
    metric_names = list(METRICS) if metric_names is None else list(metric_names)
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}

    point = {m: {nm: METRICS[m](y, preds[nm]) for nm in names} for m in metric_names}
    boot = {m: {nm: [] for nm in names} for m in metric_names}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    bdelta = {m: {f"{a}_vs_{b}": [] for a, b in pairs} for m in metric_names}

    for _ in range(B):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        yb = y[idx]
        for m in metric_names:
            vals = {}
            for nm in names:
                v = METRICS[m](yb, preds[nm][idx])
                vals[nm] = v
                boot[m][nm].append(v)
            for a, b in pairs:
                d = vals[a] - vals[b]
                bdelta[m][f"{a}_vs_{b}"].append(d)

    out = {}
    for m in metric_names:
        ci = {}
        for nm in names:
            arr = np.array(boot[m][nm], float); arr = arr[np.isfinite(arr)]
            ci[nm] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr.size else [np.nan, np.nan]
        delta = {}
        for a, b in pairs:
            arr = np.array(bdelta[m][f"{a}_vs_{b}"], float); arr = arr[np.isfinite(arr)]
            if arr.size:
                lo, hi = np.percentile(arr, [2.5, 97.5])
                # two-sided bootstrap p with +1 guard
                p = 2.0 * min((arr > 0).mean(), (arr < 0).mean())
                p = min(1.0, p + 1.0 / (arr.size + 1))
                delta[f"{a}_vs_{b}"] = dict(delta=float(point[m][a] - point[m][b]),
                                            ci=[float(lo), float(hi)], p=float(p))
            else:
                delta[f"{a}_vs_{b}"] = dict(delta=np.nan, ci=[np.nan, np.nan], p=np.nan)
        out[m] = dict(point={k: float(v) for k, v in point[m].items()}, ci=ci, delta=delta)
    out["_B"] = B
    out["_n_clusters"] = int(len(uniq))
    out["_n_rows"] = int(len(y))
    return out


# ----------------------------------------------------------------------------- DeLong (fast, paired)
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def _fast_delong(predictions_sorted, n_pos):
    m, n = n_pos, predictions_sorted.shape[1] - n_pos
    pos = predictions_sorted[:, :m]; neg = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y, p1, p2):
    """Paired DeLong test for AUROC(p1) - AUROC(p2). Returns (auc1, auc2, z, p)."""
    y = np.asarray(y); order = np.argsort(-y)  # positives first
    label_1 = y[order]; n_pos = int(label_1.sum())
    preds = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(preds, n_pos)
    l = np.array([[1, -1]])
    var = float(np.asarray(l @ cov @ l.T).reshape(-1)[0])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, np.nan
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * sps.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


# ----------------------------------------------------------------------------- multiplicity
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (returns array aligned to input)."""
    p = np.asarray(pvals, float); n = p.size
    order = np.argsort(p); ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    return out


# ----------------------------------------------------------------------------- calibration
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def spiegelhalter_z(y, p):
    """Spiegelhalter's z-test of calibration. H0: well calibrated."""
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p)))
    if den == 0:
        return np.nan, np.nan
    z = float(num / den)
    return z, float(2 * sps.norm.sf(abs(z)))


def cox_calibration(y, p):
    """Cox calibration: fit y ~ a + b*logit(p). Ideal (a,b)=(0,1)."""
    lp = _logit(p).reshape(-1, 1)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (p >= lo) & (p < hi)
        if msk.sum():
            e += msk.mean() * abs(y[msk].mean() - p[msk].mean())
    return float(e)


def calibration_block(y, p, clusters, B=2000, seed=0):
    y = np.asarray(y); p = np.asarray(p); clusters = np.asarray(clusters)
    uniq = np.unique(clusters); members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    a0, b0 = cox_calibration(y, p)
    brier0, ece0 = _brier(y, p), ece(y, p)
    aa, bb, br, ec = [], [], [], []
    for _ in range(B):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            a, b = cox_calibration(yb, pb)
            aa.append(a); bb.append(b)
        except Exception:
            pass
        br.append(_brier(yb, pb)); ec.append(ece(yb, pb))
    def ci(v):
        v = np.array(v, float); v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [np.nan, np.nan]
    sz, sp = spiegelhalter_z(y, p)
    return dict(cox_intercept=a0, cox_intercept_ci=ci(aa),
                cox_slope=b0, cox_slope_ci=ci(bb),
                brier=brier0, brier_ci=ci(br),
                ece=ece0, ece_ci=ci(ec),
                spiegelhalter_z=sz, spiegelhalter_p=sp)


# ----------------------------------------------------------------------------- per-unit AUROC vs chance
def auroc_ci_vs_chance(y, p, year_clusters, B=2000, seed=0):
    """Cluster-bootstrap (over years) CI for one unit's AUROC; flag if it clears 0.5."""
    y = np.asarray(y); p = np.asarray(p); yc = np.asarray(year_clusters)
    uniq = np.unique(yc); members = {c: np.flatnonzero(yc == c) for c in uniq}
    rng = np.random.default_rng(seed)
    point = _auroc(y, p); vals = []
    for _ in range(B):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        vals.append(_auroc(y[idx], p[idx]))
    vals = np.array(vals, float); vals = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals.size else (np.nan, np.nan))
    return dict(auroc=float(point), ci=[float(lo), float(hi)],
                n_year_clusters=int(len(uniq)),
                beats_chance=bool(np.isfinite(lo) and lo > 0.5))


# ----------------------------------------------------------------------------- climate gradient (exploratory)
def kruskal_gradient(values, groups):
    """Kruskal-Wallis across climate types. Exploratory (moderator has measurement error)."""
    by = {}
    for v, g in zip(values, groups):
        if np.isfinite(v):
            by.setdefault(str(g), []).append(v)
    groups_present = {k: vv for k, vv in by.items() if len(vv) >= 2}
    if len(groups_present) < 2:
        return dict(test="kruskal", H=np.nan, p=np.nan, groups={k: len(v) for k, v in by.items()})
    H, p = sps.kruskal(*groups_present.values())
    return dict(test="kruskal", H=float(H), p=float(p),
                group_means={k: float(np.mean(v)) for k, v in by.items()},
                group_n={k: len(v) for k, v in by.items()})
