"""
catch22_features.py — canonical time-series characteristics (catch22) as CAUSAL,
rolling features for the SparseFE pipeline.

For each week t, catch22 is computed over the trailing `window` weeks within the
same unit (ending at t, inclusive) — so it uses only information available by
week t. Two series are characterised: cases and rainfall.

Backend (honest, like the rest of the pipeline):
  - uses the real `pycatch22` package (the 22 hctsa-derived features) if installed;
  - otherwise falls back to a clearly-labelled numpy subset of robust TS features.
The method actually used is returned so it can be recorded in metrics.json.
Columns are prefixed `c22_` (real) or `c22fb_` (fallback) and tagged family
"catch22" by enrich.family_of, so they form their own rung in the comparison.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOW = 52
MIN_PERIODS = 12


def _get_catch22():
    try:
        import pycatch22
        names = pycatch22.catch22_all([0.0] * 20)["names"]
        def fn(x):
            return dict(zip(names, pycatch22.catch22_all(list(map(float, x)))["values"]))
        return "pycatch22", ["c22_" + n for n in names], fn
    except Exception:
        names = ["mean", "std", "slope", "ac1", "acf_first_zero", "prop_above_mean",
                 "longest_run_above_mean", "n_local_max", "iqr", "skew",
                 "trend_resid_std", "recent_over_old"]

        def fn(x):
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            n = len(x)
            if n < 3:
                return {k: 0.0 for k in names}
            mean = float(np.mean(x)); std = float(np.std(x))
            t = np.arange(n)
            slope = float(np.polyfit(t, x, 1)[0]) if n >= 2 else 0.0
            xc = x - mean
            ac1 = float(np.dot(xc[1:], xc[:-1]) / (np.dot(xc, xc) + 1e-12))
            # first zero crossing of the autocorrelation function
            acf_zero = 0
            denom = np.dot(xc, xc) + 1e-12
            for lag in range(1, n):
                if np.dot(xc[lag:], xc[:-lag]) / denom <= 0:
                    acf_zero = lag; break
            above = x > mean
            prop_above = float(np.mean(above))
            # longest consecutive run above the mean
            longest = cur = 0
            for a in above:
                cur = cur + 1 if a else 0
                longest = max(longest, cur)
            n_local_max = int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))) if n >= 3 else 0
            iqr = float(np.subtract(*np.percentile(x, [75, 25])))
            skew = float(np.mean(xc ** 3) / (std ** 3 + 1e-12))
            resid = x - (slope * t + (mean - slope * t.mean()))
            trend_resid_std = float(np.std(resid))
            half = max(1, n // 2)
            recent_over_old = float((np.mean(x[-half:]) + 1) / (np.mean(x[:half]) + 1))
            return dict(mean=mean, std=std, slope=slope, ac1=ac1,
                        acf_first_zero=float(acf_zero), prop_above_mean=prop_above,
                        longest_run_above_mean=float(longest), n_local_max=float(n_local_max),
                        iqr=iqr, skew=skew, trend_resid_std=trend_resid_std,
                        recent_over_old=recent_over_old)
        return "numpy_fallback", ["c22fb_" + n for n in names], fn


def _rolling_apply(series: np.ndarray, fn, prefix_cols, raw_names, window, min_periods):
    n = len(series)
    out = {c: np.full(n, np.nan) for c in prefix_cols}
    for t in range(n):
        lo = max(0, t - window + 1)
        w = series[lo:t + 1]
        w = w[np.isfinite(w)]
        if len(w) < min_periods:
            continue
        feats = fn(w)
        for c, rn in zip(prefix_cols, raw_names):
            out[c][t] = feats.get(rn, np.nan)
    return out


def add_catch22(df: pd.DataFrame, case_col: str, rain_col: str, group_cols: list[str],
                window: int = WINDOW, min_periods: int = MIN_PERIODS):
    """Append causal rolling catch22 features for cases and rainfall.
    Returns (df_with_features, method_name, list_of_added_columns)."""
    method, cols, fn = _get_catch22()
    raw_names = [c.split("_", 1)[1] for c in cols]  # strip c22_/c22fb_ prefix
    df = df.sort_values(group_cols + ["YR", "WN"]).reset_index(drop=True)
    case_cols = [c + "_case" for c in cols]
    rain_cols = [c + "_rain" for c in cols]
    for c in case_cols + rain_cols:
        df[c] = np.nan
    keys = group_cols if group_cols else None
    groups = [(None, df)] if keys is None else df.groupby(keys, sort=False)
    for _, idx in (groups.groups.items() if keys else [(None, df.index)]):
        sub = df.loc[idx] if keys else df
        cs = _rolling_apply(sub[case_col].to_numpy(float), fn, case_cols, raw_names, window, min_periods)
        rs = _rolling_apply(sub[rain_col].to_numpy(float), fn, rain_cols, raw_names, window, min_periods)
        for c, v in {**cs, **rs}.items():
            df.loc[sub.index, c] = v
    added = case_cols + rain_cols
    return df, method, added
