"""
labels.py — early-warning target.

Two label definitions are available via the LABEL_MODE environment variable:

  LABEL_MODE=limb   (default)  ascending-limb target.
      Positive on the rising limb from the pre-peak trough (onset) up to the
      annual peak; NEGATIVE everywhere else in the year, including the
      descending limb and the troughs. Evaluated against the whole year. Raw
      case level cannot separate rising from falling weeks at the same height,
      so the two raw columns are genuinely weak here and featurisation has room
      to help. This matches the ascending-limb design of the companion writeup.

  LABEL_MODE=prepeak           the older target: positive in the H weeks
      immediately before the peak, evaluated only on pre-peak weeks. Raw case
      level nearly separates the classes, so raw scores high (useful as a
      contrast, not as the headline target).

Both are case-derived, so achievable AUROC is bounded; features only ever use
information up to week t.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HORIZON = 8  # prepeak mode: weeks before the peak counted as positive


def _limb(sub, cases):
    """Positive on onset..peak (rising limb); all weeks evaluated."""
    n = len(cases)
    peak = int(np.nanargmax(cases))
    pre = cases[:peak + 1]
    onset = int(np.nanargmin(pre)) if peak > 0 else 0   # trough before the peak
    wk = np.arange(n)
    sub["y"] = ((wk >= onset) & (wk < peak)).astype(int)  # ascending limb, pre-peak
    sub["labelable"] = 1                                   # rest of the year = negatives
    return sub


def _prepeak(sub, cases, horizon):
    peak = int(np.nanargmax(cases))
    wk = np.arange(len(cases))
    lead = peak - wk
    sub["y"] = ((lead > 0) & (lead <= horizon)).astype(int)
    sub["labelable"] = (wk < peak).astype(int)
    return sub


def build_peak_labels(df: pd.DataFrame, case_col: str, group_cols: list[str],
                      horizon: int = HORIZON) -> pd.DataFrame:
    mode = os.environ.get("LABEL_MODE", "limb").lower()
    out = []
    keys = group_cols + ["YR"]
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.sort_values("WN").copy()
        cases = sub[case_col].to_numpy(dtype=float)
        if len(cases) < 10 or np.all(np.isnan(cases)) or np.nanmax(cases) <= 0:
            sub["y"] = 0; sub["labelable"] = 0
            out.append(sub); continue
        out.append(_limb(sub, cases) if mode == "limb" else _prepeak(sub, cases, horizon))
    return pd.concat(out, ignore_index=True)
