"""
evaluate.py — leave-one-year-out (expanding) evaluation with TabPFN.

Backend selection (honest + Windows-friendly):
  AUTOTABPFN_BACKEND=tabpfn  -> require TabPFN (set TABPFN_TOKEN to skip the
                                browser login that crashes on Windows)
  AUTOTABPFN_BACKEND=standin -> deterministic offline model (no account needed)
  unset / "auto"             -> try TabPFN, fall back to the stand-in
Every result records which backend ran. Stand-in numbers are NOT TabPFN numbers.

Imputer + rank-Gaussian (QuantileTransformer) + the classifier are all fit on
TRAIN years only, then applied to the held-out year, so nothing leaks.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RNG = 0


def get_backend():
    want = os.environ.get("AUTOTABPFN_BACKEND", "auto").lower()
    n_est = int(os.environ.get("N_EST", "4"))
    if want in ("auto", "tabpfn"):
        try:
            from tabpfn import TabPFNClassifier
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
            def fp(Xtr, ytr, Xte):
                clf = TabPFNClassifier(n_estimators=n_est, device="cpu",
                                       balance_probabilities=True, random_state=RNG)
                clf.fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)
                return proba[:, 1] if proba.shape[1] > 1 else np.full(len(Xte), float(clf.classes_[0]))
            return "tabpfn", fp
        except Exception:
            if want == "tabpfn":
                raise
    from sklearn.ensemble import HistGradientBoostingClassifier
    def fp(Xtr, ytr, Xte):
        m = HistGradientBoostingClassifier(random_state=RNG, class_weight="balanced")
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)
        return proba[:, 1] if proba.shape[1] > 1 else np.full(len(Xte), float(m.classes_[0]))
    return "standin_histgb", fp


def _cap_train(tr, seed=RNG):
    """Subsample the in-context training set to <= TABPFN_MAX_TRAIN rows while
    ALWAYS keeping both classes. Positives are rare, so we keep up to half the
    budget for them and fill the rest with negatives (and vice-versa)."""
    max_train = int(os.environ.get("TABPFN_MAX_TRAIN", "1000"))
    if len(tr) <= max_train:
        return tr
    pos = tr[tr["y"] == 1]
    neg = tr[tr["y"] == 0]
    # reserve room for both classes
    n_pos = min(len(pos), max(1, max_train // 2))
    n_neg = min(len(neg), max_train - n_pos)
    n_pos = min(len(pos), max_train - n_neg)   # give leftover back to positives
    out = pd.concat([pos.sample(n_pos, random_state=seed),
                     neg.sample(n_neg, random_state=seed)])
    return out.sample(frac=1, random_state=seed)


def _prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    a, b = imp.transform(Xtr), imp.transform(Xte)
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(1000, len(a)), random_state=RNG).fit(a)
    return qt.transform(a), qt.transform(b)


def loyo(df: pd.DataFrame, cols: list[str], backend_fp, return_oof: bool = False):
    df = df[df["labelable"] == 1].copy()
    years = sorted(df["YR"].unique())
    Y, P = [], []
    for y in years[2:]:                       # expanding: need >=2 train years
        tr, te = df[df.YR < y], df[df.YR == y]
        if tr["y"].sum() == 0 or len(te) == 0 or te["y"].nunique() < 1:
            continue
        tr = _cap_train(tr)                   # TabPFN CPU limit + small-context design
        ytr = tr["y"].to_numpy()
        if len(np.unique(ytr)) < 2:           # safety: never fit on a single class
            continue
        Atr, Ate = _prep(tr[cols].to_numpy(), te[cols].to_numpy())
        p_te = backend_fp(Atr, ytr, Ate)
        if os.environ.get("AUTOTABPFN_CALIBRATE", "1") == "1":
            Atr2, _ = _prep(tr[cols].to_numpy(), tr[cols].to_numpy())
            p_tr = backend_fp(Atr2, ytr, Atr2)   # second fit only when calibrating
            try:
                p_te = IsotonicRegression(out_of_bounds="clip").fit(p_tr, ytr).predict(p_te)
            except Exception:
                pass
        Y.append(te["y"].to_numpy()); P.append(p_te)
    Y, P = np.concatenate(Y), np.concatenate(P)
    res = dict(n_features=len(cols), n_test=int(len(Y)), base_rate=float(Y.mean()),
               auroc=float(roc_auc_score(Y, P)),
               auprc=float(average_precision_score(Y, P)),
               brier=float(brier_score_loss(Y, P)))
    if return_oof:
        return res, Y, P
    return res


def loyo_per_region(df: pd.DataFrame, cols: list[str], group_col: str, backend_fp,
                    climate_map: dict | None = None) -> pd.DataFrame:
    """Per-unit leave-one-year-out metrics, with optional climate-type tag."""
    rows = []
    for unit, sub in df.groupby(group_col):
        try:
            r = loyo(sub, cols, backend_fp)
        except Exception:
            continue
        r[group_col] = unit
        if climate_map is not None:
            r["climate_type"] = climate_map.get(str(unit), "NA")
        rows.append(r)
    return pd.DataFrame(rows)
