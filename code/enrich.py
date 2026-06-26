"""
enrich.py — THE METHOD.

Turn two raw weekly columns (cases, rainfall) into ~100 domain-specific,
leakage-safe features that expose the latent structure a tabular foundation
model can read: seasonality, accumulation/threshold effects, transmission
acceleration, critical slowing down, and low-frequency climate (ENSO-like)
signal derived from rainfall itself.

Every feature at week t uses only weeks <= t within the same (unit, year), and
week-of-year climatology uses prior years only (expanding, shifted). No future.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- STA/LTA transmission-acceleration detector (fixed, established) -------
def _stalta_year(cases):
    n = len(cases); out = np.full(n, np.nan)
    for t in range(n):
        sta = np.nanmean(cases[max(0, t - 3): t + 1])
        hi = t - 5
        if hi <= 0:
            continue
        lo = max(0, hi - 26)
        seg = cases[lo:hi]
        if len(seg) < 4:
            continue
        lta = np.nanmean(seg)
        out[t] = sta / lta if lta and lta > 0 else np.nan
    return out


def _g(df, keys):
    return [df[k] for k in keys]


def enrich(df: pd.DataFrame, case_col: str, rain_col: str,
           group_cols: list[str]) -> pd.DataFrame:
    """Return df with YR, WN, group cols, raw two columns, and ~100 features."""
    df = df.sort_values(group_cols + ["YR", "WN"]).reset_index(drop=True)
    keys = group_cols + ["YR"]
    F = pd.DataFrame(index=df.index)
    c = df[case_col].astype(float)
    r = df[rain_col].astype(float)
    gy = df.groupby(keys, sort=False)

    def roll(s, w, fn):
        return s.groupby(_g(df, keys)).transform(lambda x: getattr(x.rolling(w, min_periods=1), fn)())

    def rollvar(s, w):
        return s.groupby(_g(df, keys)).transform(lambda x: x.rolling(w, min_periods=3).var())

    def rollac1(s, w):
        return s.groupby(_g(df, keys)).transform(
            lambda x: x.rolling(w, min_periods=4).apply(
                lambda v: pd.Series(v).autocorr(1) if np.nanstd(v) > 0 else 0.0, raw=False))

    def stalta(s):
        out = pd.Series(np.nan, index=df.index)
        for _, idx in gy.groups.items():
            sub = df.loc[idx].sort_values("WN")
            out.loc[sub.index] = _stalta_year(s.loc[sub.index].to_numpy())
        return out

    # 1) raw + lags (cases, rainfall)  ~16
    for L in (1, 2, 3, 4, 6, 8):
        F[f"case_lag{L}"] = c.groupby(_g(df, keys)).shift(L)
        F[f"rain_lag{L}"] = r.groupby(_g(df, keys)).shift(L)
    F["case_raw"] = c
    F["rain_raw"] = r

    # 2) rolling means / sums / std  ~24
    for w in (2, 4, 8, 12, 26):
        F[f"case_mean{w}"] = roll(c, w, "mean")
        F[f"rain_mean{w}"] = roll(r, w, "mean")
        F[f"rain_sum{w}"] = roll(r, w, "sum")
    for w in (4, 8, 12):
        F[f"case_std{w}"] = roll(c, w, "std")
        F[f"rain_std{w}"] = roll(r, w, "std")

    # 3) accumulation windows for rainfall (threshold/habitat persistence)  ~6
    for w in (4, 6, 8, 10, 12, 16):
        F[f"rain_accum{w}"] = roll(r, w, "sum")

    # 4) transmission acceleration (order parameter) on cases & rainfall  ~2
    F["case_stalta"] = stalta(c)
    F["rain_stalta"] = stalta(r)

    # 5) momentum: deltas & acceleration  ~6
    for name, s in (("case", c), ("rain", r)):
        d = s.groupby(_g(df, keys)).diff()
        F[f"{name}_delta"] = d
        F[f"{name}_accel"] = d.groupby(_g(df, keys)).diff()
        F[f"{name}_pct"] = s.groupby(_g(df, keys)).pct_change().replace([np.inf, -np.inf], np.nan)

    # 6) week-of-year climatology (prior years only) + anomalies + z  ~6
    woy_keys = group_cols + ["WN"]
    for name, s in (("case", c), ("rain", r)):
        gm = s.groupby(_g(df, woy_keys)).transform(lambda x: x.expanding().mean().shift(1))
        gs = s.groupby(_g(df, woy_keys)).transform(lambda x: x.expanding().std().shift(1))
        F[f"{name}_woy_anom"] = (s - gm)
        F[f"{name}_woy_z"] = (s - gm) / gs.replace(0, np.nan)
        F[f"{name}_woy_mean"] = gm

    # 7) Fourier seasonality (explicit periodicity)  ~6
    woy = df["WN"].astype(float)
    for k in (1, 2, 3):
        F[f"sin{k}"] = np.sin(2 * np.pi * k * woy / 52.0)
        F[f"cos{k}"] = np.cos(2 * np.pi * k * woy / 52.0)

    # 8) phase-transition views of 8-week rainfall at several critical scales  ~5
    r8 = roll(r, 8, "sum")
    for Rc in (250, 350, 450, 550):
        z = np.clip(-0.02 * (r8 - Rc), -50, 50)
        F[f"rain_logistic_Rc{Rc}"] = 1.0 / (1.0 + np.exp(z))
    F["rain_hill_R8"] = r8**2 / (r8**2 + 400.0**2)

    # 9) ENSO-like / monsoon proxies from rainfall low-frequency structure  ~7
    F["rain_long26_anom"] = roll(r, 26, "mean") - roll(r, 52, "mean")
    F["rain_long52_mean"] = roll(r, 52, "mean")
    F["rain_12w_minus_woy"] = roll(r, 12, "mean") - F["rain_woy_mean"]
    wet = (roll(r, 8, "sum") > roll(r, 26, "mean") * 8).astype(float)
    F["wet_season_flag"] = wet
    F["wet_run_len"] = wet.groupby(_g(df, keys)).transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1) * wet
    F["rain_cumyear"] = r.groupby(_g(df, keys)).cumsum()
    F["weeks_into_year"] = gy.cumcount()

    # 10) critical slowing down (early-warning of transition)  ~4
    F["case_var12"] = rollvar(c, 12)
    F["rain_var12"] = rollvar(r, 12)
    F["case_ac1_12"] = rollac1(c, 12)
    F["rain_ac1_12"] = rollac1(r, 12)

    # 11) interactions: climate x season, climate x state  ~4
    F["rain8_x_sin1"] = r8 * F["sin1"]
    F["rain8_x_cos1"] = r8 * F["cos1"]
    F["rain8_x_caseStalta"] = r8 * F["case_stalta"].fillna(0)
    F["caseLag1_x_rain8"] = F["case_lag1"].fillna(0) * r8

    # 12) extra views to reach ~100 columns  ~25
    for L in (10, 12, 16):
        F[f"case_lag{L}"] = c.groupby(_g(df, keys)).shift(L)
        F[f"rain_lag{L}"] = r.groupby(_g(df, keys)).shift(L)
    for w in (4, 12):
        F[f"case_max{w}"] = roll(c, w, "max")
        F[f"case_min{w}"] = roll(c, w, "min")
        F[f"rain_max{w}"] = roll(r, w, "max")
    F["log_case"] = np.log1p(c.clip(lower=0))
    F["log_rain"] = np.log1p(r.clip(lower=0))
    F["case_over_rain8"] = c / (r8 + 1.0)
    F["rain8_over_woy"] = r8 / (F["rain_woy_mean"].abs() + 1.0)
    F["rain_pct_unit"] = (roll(r, 8, "sum")
                          .groupby(df[group_cols[0]] if group_cols else pd.Series("ALL", index=df.index))
                          .rank(pct=True))
    F["stalta_delta"] = F["case_stalta"].groupby(_g(df, keys)).diff()
    F["rain_sum8_woy_anom"] = r8 - r8.groupby(_g(df, woy_keys)).transform(
        lambda x: x.expanding().mean().shift(1))
    F["case_cumyear"] = c.groupby(_g(df, keys)).cumsum()
    F["rain_skew12"] = r.groupby(_g(df, keys)).transform(
        lambda x: x.rolling(12, min_periods=4).skew())
    F["case_skew12"] = c.groupby(_g(df, keys)).transform(
        lambda x: x.rolling(12, min_periods=4).skew())
    F["ix_enso_proxy"] = roll(r, 26, "mean") - F["rain_long52_mean"]
    F["ix_rain_anom_ratio"] = F["rain_woy_anom"] / (F["rain_woy_mean"].abs() + 1.0)

    # 13) epidemic memory / immunity  (multi-year, within unit, strictly causal)
    def _weeks_since_high(cases):
        # weeks since incidence last reached epidemic level, where the threshold is
        # the expanding 90th percentile of the unit's OWN prior history (no future info).
        n = len(cases); out = np.full(n, np.nan); last = -1; hist = []
        for t in range(n):
            out[t] = float(t - last) if last >= 0 else float(t)
            v = cases[t]
            if hist:
                thr = np.nanpercentile(hist, 90)
                if np.isfinite(v) and np.isfinite(thr) and v >= thr:
                    last = t
            if np.isfinite(v):
                hist.append(v)
        return out

    iei = pd.Series(np.nan, index=df.index)
    if group_cols:
        groups = df.groupby(group_cols, sort=False).groups.items()
        c52 = c.groupby(_g(df, group_cols)).transform(
            lambda x: x.rolling(52, min_periods=8).sum())
    else:
        groups = [("ALL", df.index)]
        c52 = c.rolling(52, min_periods=8).sum()
    for _, idx in groups:
        sub = df.loc[idx].sort_values(["YR", "WN"])
        iei.loc[sub.index] = _weeks_since_high(c.loc[sub.index].to_numpy())
    F["inter_epidemic_interval"] = iei                      # susceptibility clock
    F["case_sum52"] = c52                                   # recent burden / immunity depletion
    F["susceptibility_proxy"] = iei / (np.log1p(c52.clip(lower=0)) + 1.0)

    meta = df[group_cols + ["YR", "WN", case_col, rain_col]].copy()
    out = pd.concat([meta, F.replace([np.inf, -np.inf], np.nan)], axis=1)
    return out


RAW_TWO = ["case_raw", "rain_raw"]  # the unenriched 2-column baseline


def feature_columns(enriched: pd.DataFrame, case_col, rain_col, group_cols):
    drop = set(group_cols + ["YR", "WN", case_col, rain_col, "y", "labelable"])
    return [c for c in enriched.columns if c not in drop]


def classify_column(name: str) -> str:
    """Tag a feature as catch22, rain-derived, case-derived, or calendar/seasonal."""
    n = name.lower()
    if n.startswith("c22"):
        return "catch22"
    if any(k in n for k in ("rain", "wet", "monsoon", "enso", "ix_", "logistic", "hill", "_rc")):
        return "rain"
    if n.startswith(("sin", "cos", "weeks_into")):
        return "calendar"
    if "case" in n or "stalta" in n or "log_case" in n:
        return "case"
    return "case"


def case_calendar_columns(cols: list[str]) -> list[str]:
    """The 'case-only' rung: case + calendar features, rainfall removed."""
    return [c for c in cols if classify_column(c) in ("case", "calendar")]


_RATIONALE = {
    "case_lag": "lagged case count — recent transmission level",
    "rain_lag": "lagged rainfall — delayed vector-habitat effect",
    "case_mean": "smoothed case level over the window",
    "rain_mean": "smoothed rainfall over the window",
    "rain_sum": "accumulated rainfall — habitat persistence",
    "rain_accum": "n-week accumulated rainfall — threshold/breeding signal",
    "case_std": "case volatility over the window",
    "rain_std": "rainfall variability over the window",
    "case_stalta": "transmission-acceleration order parameter (cases)",
    "rain_stalta": "rainfall surge relative to baseline",
    "inter_epidemic_interval": "weeks since last epidemic-level incidence (susceptibility clock)",
    "susceptibility_proxy": "accumulated-susceptibles proxy (long interval, low recent burden)",
    "sum52": "52-week cumulative cases (recent burden / immunity depletion)",
    "case_delta": "week-on-week change in cases",
    "rain_delta": "week-on-week change in rainfall",
    "case_accel": "acceleration of cases",
    "rain_accel": "acceleration of rainfall",
    "case_pct": "relative change in cases",
    "rain_pct": "relative change in rainfall",
    "woy_anom": "anomaly vs prior-years week-of-year climatology",
    "woy_z": "standardized week-of-year anomaly",
    "woy_mean": "prior-years week-of-year baseline",
    "sin": "Fourier seasonality term", "cos": "Fourier seasonality term",
    "logistic_Rc": "phase-transition view of 8-week rainfall at a critical scale",
    "hill_R8": "saturating (Hill) view of 8-week rainfall",
    "long26": "low-frequency rainfall anomaly (ENSO-like)",
    "long52": "annual-scale rainfall level (ENSO-like)",
    "wet_season": "wet-season indicator from rainfall structure",
    "wet_run": "consecutive wet weeks",
    "cumyear": "cumulative within-year total",
    "weeks_into_year": "calendar position in the year",
    "var12": "rolling variance — critical slowing down",
    "ac1_12": "rolling lag-1 autocorrelation — critical slowing down",
    "x_": "interaction term", "_x_": "interaction term",
    "log_": "log1p-compressed level",
    "over_": "ratio feature", "_over_": "ratio feature",
    "pct_unit": "within-unit percentile rank of accumulated rainfall",
    "skew": "rolling skewness", "max": "rolling maximum", "min": "rolling minimum",
    "ix_": "climate index proxy derived from rainfall",
}


def feature_dictionary(cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        rationale = next((v for k, v in _RATIONALE.items() if k in c),
                         "domain-derived feature")
        rows.append(dict(feature=c, source=classify_column(c), rationale=rationale))
    return pd.DataFrame(rows)


def _is_biology(c):
    return (("stalta" in c) or ("var12" in c) or ("ac1_" in c) or ("_x_" in c)
            or ("inter_epidemic" in c) or ("suscept" in c) or ("sum52" in c))


def ladder_groups(cols: list[str]) -> dict:
    """Split features into the enrichment-ladder rungs used in fig1b:
    case views -> climate -> transmission biology."""
    biology = [c for c in cols if _is_biology(c)]
    case_view = [c for c in cols if (not _is_biology(c)) and classify_column(c) == "case"]
    climate = [c for c in cols if (not _is_biology(c)) and classify_column(c) in ("rain", "calendar")]
    return dict(case_view=case_view, climate=climate, biology=biology)


def family_of(c: str) -> str:
    """Coarse family tag for the attribution figure colors."""
    cl = c.lower()
    if cl.startswith("c22"):
        return "catch22"
    if _is_biology(c):
        return "transmission" if any(k in cl for k in ("stalta", "inter_epidemic", "suscept", "sum52")) else "early-warning"
    if classify_column(c) == "rain":
        return "climate"
    if classify_column(c) == "calendar":
        return "season"
    return "case"
