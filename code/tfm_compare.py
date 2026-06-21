"""
tfm_compare.py — time-series foundation-model comparator for fig1 panel a.

The point of this bar: show that a *time-series* model reading raw cases alone
underperforms TabPFN reading the enriched representation — i.e. representation
beats model. It forecasts cases from recent history and scores each week by the
expected upcoming rise, then reports AUROC/AUPRC against the pre-peak label.

Backend (honest + Windows-friendly):
  uses TimesFM if it imports and loads; otherwise falls back to a classical
  linear-trend forecaster, and the returned name says which one ran. A
  `ts_trend_baseline` bar is NOT a TimesFM result and is labelled as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

MIN_HISTORY = 12
HORIZON = 8


def get_forecaster():
    try:
        import timesfm  # best-effort; heavy, separate stack, often unavailable
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.5-200m-pytorch"),
        )
        def forecast(hist):
            pt, _ = tfm.forecast([np.asarray(hist, dtype=float)], freq=[0])
            return np.asarray(pt[0])
        return "timesfm", forecast
    except Exception:
        def forecast(hist):
            h = np.asarray(hist, dtype=float)
            w = min(len(h), 8)
            x = np.arange(w); yv = h[-w:]
            slope = np.polyfit(x, yv, 1)[0] if w >= 2 else 0.0
            return h[-1] + slope * np.arange(1, HORIZON + 1)
        return "ts_trend_baseline", forecast


def tfm_score(df: pd.DataFrame, case_col: str, group_cols: list[str]) -> dict:
    name, forecast = get_forecaster()
    keys = group_cols + ["YR"]
    ys, scores = [], []
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.sort_values("WN")
        cases = sub[case_col].to_numpy(dtype=float)
        lab = sub["labelable"].to_numpy()
        y = sub["y"].to_numpy()
        for t in range(len(sub)):
            if not lab[t]:
                continue
            hist = cases[: t + 1]
            if len(hist) < MIN_HISTORY:
                continue
            f = forecast(hist[-104:])
            rise = float(np.nanmax(f) - hist[-1])      # expected upcoming increase
            ys.append(int(y[t])); scores.append(rise)
    ys, scores = np.array(ys), np.array(scores)
    if len(np.unique(ys)) < 2:
        return dict(name=name, auroc=float("nan"), auprc=float("nan"), n=int(len(ys)))
    return dict(name=name, auroc=float(roc_auc_score(ys, scores)),
                auprc=float(average_precision_score(ys, scores)), n=int(len(ys)))
