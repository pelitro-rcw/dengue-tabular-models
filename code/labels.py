"""
labels.py — peak-anchored early-warning label.

The outcome is "will the annual epidemic peak arrive within the next H weeks".
Positive in the H-week rising limb before each unit-year's case peak. The label
is case-derived (this is honest: it caps achievable AUROC ~0.85, as the
manuscript notes), but features only ever use information up to week t.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HORIZON = 8  # weeks before the peak counted as positive


def build_peak_labels(df: pd.DataFrame, case_col: str, group_cols: list[str],
                      horizon: int = HORIZON) -> pd.DataFrame:
    out = []
    keys = group_cols + ["YR"]
    for _, sub in df.groupby(keys, sort=False):
        sub = sub.sort_values("WN").copy()
        cases = sub[case_col].to_numpy(dtype=float)
        if len(cases) < 10 or np.all(np.isnan(cases)) or np.nanmax(cases) <= 0:
            sub["y"] = 0; sub["labelable"] = 0
            out.append(sub); continue
        peak = int(np.nanargmax(cases))
        wk = np.arange(len(sub))
        lead = peak - wk
        sub["y"] = ((lead > 0) & (lead <= horizon)).astype(int)
        sub["labelable"] = (wk < peak).astype(int)  # only weeks before the peak
        out.append(sub)
    return pd.concat(out, ignore_index=True)
