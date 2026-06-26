"""
rain_threshold.py — rainfall trigger that PRECEDES the TA ONSET, derived with
Youden's J (sensitivity / specificity / AUROC), an in-fold search over rainfall
REPRESENTATION + lead, and leave-one-year-out validation.

Why a representation search: a raw accumulation LEVEL is high all wet season and
cannot isolate the pre-onset weeks (low specificity). The onset follows the
dry->wet TRANSITION, which is carried by rainfall DYNAMICS. The search therefore
chooses, inside each fold, among three families:
  * accumulation  : W-week rolling rainfall sum            (accum4/8/12)
  * rate-of-rise  : accumW minus accumW W weeks earlier    (rate4/8)
  * rainfall STA/LTA : short MA / long MA of weekly rain   (stalta3_12 / stalta4_26)
On QC/regional this lifts onset AUROC from ~0.5-0.6 (level) to ~0.76-0.81
(STA/LTA), organically, because a sharp rainfall rise is specific to the
transition that precedes acceleration.

TARGET : positive week = the TA ONSET (first Constant-TA trigger of the year,
         Stage 3-5 STA/LTA detector) falls within the next L weeks.
SELECT : (representation, lead, direction, threshold) maximising Youden J,
         chosen on training years; held-out year tested. Rotated over years.

Outputs: outputs/<scale>/rain_thresholds.csv
Env: ACCUM_WINDOWS ("4,8,12"), LEADS ("4,6,8,10"), MIN_LEAD ("4"),
     TA_MODE (constant|continuous), ETA_ON (1.33), ETA_OFF (0.73)
Usage: py code/rain_threshold.py --scale qc
"""
import os, argparse
from collections import Counter
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCUM_WINDOWS = [int(x) for x in os.environ.get("ACCUM_WINDOWS", "4,8,12").split(",")]
RATE_WINDOWS = [4, 8]
STALTA_PAIRS = [(3, 12), (4, 26)]
LEADS = [int(x) for x in os.environ.get("LEADS", "4,6,8,10").split(",")]
MIN_LEAD = int(os.environ.get("MIN_LEAD", "4"))
EFF_LEADS = [l for l in LEADS if l >= MIN_LEAD] or [MIN_LEAD]
TA_MODE = os.environ.get("TA_MODE", "constant").lower()
ETA_ON = float(os.environ.get("ETA_ON", "1.33"))
ETA_OFF = float(os.environ.get("ETA_OFF", "0.73"))
STA_WIN, LTA_WIN, GUARD, MIN_OFF_RESET = 4, 26, 2, 8
MIN_T = STA_WIN + GUARD + LTA_WIN
_BOOT_B = int(os.environ.get("BOOT_B", "2000"))


def _unit_col(cols):
    for c in ("COUNTRY", "REGION"):
        if c in cols:
            return c
    return None


def _col(cols, prefix):
    for c in cols:
        if str(c).startswith(prefix):
            return c
    return None


def _constant_ta(dc, yr):
    n = len(dc); on = np.zeros(n, dtype=bool)
    is_on = False; frozen = np.nan; coff = 0
    for i in range(n):
        if i > 0 and yr[i] != yr[i - 1]:
            is_on = False; frozen = np.nan; coff = 0
        coff = coff + 1 if not is_on else 0
        if (not is_on) and coff >= MIN_OFF_RESET:
            frozen = np.nan
        if i + 1 < MIN_T:
            continue
        sv = dc[i - STA_WIN + 1:i + 1]
        sta = np.nan if np.all(np.isnan(sv)) else np.nanmean(sv)
        if (not is_on) or np.isnan(frozen):
            lo, hi = i - 31, i - 5
            frozen = (np.nan if lo < 0 or np.all(np.isnan(dc[lo:hi])) else np.nanmean(dc[lo:hi]))
        R = sta / frozen if (not np.isnan(frozen) and frozen > 0 and not np.isnan(sta)) else np.nan
        if (not is_on) and (not np.isnan(R)) and R >= ETA_ON:
            is_on = True; coff = 0
        if is_on and (not np.isnan(R)) and R < ETA_OFF:
            is_on = False; frozen = np.nan
        on[i] = is_on
    return on.astype(int)


def _continuous_ta(dc):
    s = pd.Series(dc, dtype=float)
    return ((s.rolling(12, min_periods=1).mean() > 0) &
            (s.rolling(3, min_periods=1).mean() / s.rolling(12, min_periods=1).mean() > ETA_ON)).to_numpy().astype(int)


def _build_features(g, rc):
    """Candidate rainfall representations (causal) for one unit."""
    r = g[rc].fillna(0)
    f = {}
    for W in ACCUM_WINDOWS:
        f[f"rain_accum{W}w"] = r.rolling(W, min_periods=1).sum().to_numpy()
    for W in RATE_WINDOWS:
        a = r.rolling(W, min_periods=1).sum()
        f[f"rain_rate{W}w"] = (a - a.shift(W)).to_numpy()
    for s, l in STALTA_PAIRS:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (r.rolling(s, min_periods=1).mean() /
                     r.rolling(l, min_periods=1).mean())
        f[f"rain_stalta{s}_{l}"] = ratio.replace([np.inf, -np.inf], np.nan).to_numpy()
    return f


def _onset_target(g, L):
    yr = g["YR"].to_numpy(); wn = g["WN"].to_numpy(); ta = g["_ta"].to_numpy()
    out = np.full(len(g), np.nan)
    for y in np.unique(yr):
        idx = np.where(yr == y)[0]
        on = wn[idx][ta[idx] == 1]
        if len(on) == 0:
            continue
        lead = on.min() - wn[idx]
        out[idx] = ((lead >= 1) & (lead <= L)).astype(float)
    return out


def _auc(y, x):
    m = np.isfinite(x) & np.isfinite(y); y, x = y[m].astype(int), x[m].astype(float)
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos) / (len(pos) * len(neg)))


def _fit(x, y):
    """Best 'rain >= T' threshold by Youden J (fixed rising direction)."""
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m].astype(int)
    if len(x) < 20 or len(np.unique(y)) < 2:
        return None
    auroc = _auc(y, x)
    cand = np.unique(x)
    if len(cand) > 120:
        cand = np.quantile(x, np.linspace(0.02, 0.98, 100))
    best = None
    for t in cand:
        pred = x >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        tn = np.sum(~pred & (y == 0)); fn = np.sum(~pred & (y == 1))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if best is None or j > best[0]:
            best = (j, float(t), sens, spec)
    j, t, sens, spec = best
    return dict(youden_j=j, threshold=t, direction=">=", sensitivity=sens,
                specificity=spec, auroc=auroc, n=int(len(x)))


def _select(feats, tg, mask):
    best = None
    for fname, x in feats.items():
        xm = x[mask]
        for L in EFF_LEADS:
            r = _fit(xm, tg[L][mask])
            if r and (best is None or r["youden_j"] > best["youden_j"]):
                best = dict(feature=fname, lead=L, **r)
    return best


def _median_lead(g, x, thr, d):
    g = g.assign(_x=x); leads = []
    for _, gy in g.groupby("YR"):
        gy = gy.sort_values("WN"); on = gy.loc[gy["_ta"] == 1, "WN"]
        if not len(on):
            continue
        cr = gy.loc[(gy["_x"] >= thr) if d == ">=" else (gy["_x"] <= thr), "WN"]
        if len(cr):
            lead = int(on.min()) - int(cr.min())
            if lead > 0:
                leads.append(lead)
    return float(np.median(leads)) if leads else np.nan


def _loyo(feats, subset, tg, yr, wn=None, return_oof=False):
    years = sorted(set(yr)); fa = []; preds = []; labs = []; chosen = []; thp = []
    oof_s, oof_y, oof_yr, oof_wn = [], [], [], []
    for yt in years:
        tr = yr != yt; te = yr == yt
        sel = _select(subset, tg, tr)
        if not sel:
            continue
        xf = feats[sel["feature"]]; thf = sel["threshold"]
        xt, ylab = xf[te], tg[sel["lead"]][te]
        wt = wn[te] if wn is not None else np.arange(len(xf))[te]
        m = np.isfinite(xt) & np.isfinite(ylab); xt, ylv = xt[m], ylab[m].astype(int)
        if len(xt) == 0:
            continue
        if len(np.unique(ylv)) == 2:
            fa.append(_auc(ylv, xt))
        preds.append((xt >= thf).astype(int)); labs.append(ylv)
        chosen.append(sel["feature"]); thp.append(float(np.nanmean(xf[tr] < thf) * 100))
        if return_oof:
            oof_s.append(xt); oof_y.append(ylv)
            oof_yr.append(np.full(len(ylv), str(yt))); oof_wn.append(wt[m])
    if not preds:
        res = dict(auroc_loyo=np.nan, balanced_acc_loyo=np.nan, sens_loyo=np.nan,
                   spec_loyo=np.nan, modal_feature=None, threshold_pctl_cv=np.nan)
        return (res, dict(score=np.array([]), y=np.array([]), year=np.array([]), wn=np.array([]))) if return_oof else res
    P = np.concatenate(preds); Yl = np.concatenate(labs)
    tp = np.sum((P == 1) & (Yl == 1)); fn = np.sum((P == 0) & (Yl == 1))
    tn = np.sum((P == 0) & (Yl == 0)); fp = np.sum((P == 1) & (Yl == 0))
    se = tp / (tp + fn) if (tp + fn) else np.nan
    sp = tn / (tn + fp) if (tn + fp) else np.nan
    res = dict(auroc_loyo=round(float(np.mean(fa)), 3) if fa else np.nan,
               balanced_acc_loyo=round(float(np.nanmean([se, sp])), 3),
               sens_loyo=round(float(se), 3) if se == se else np.nan,
               spec_loyo=round(float(sp), 3) if sp == sp else np.nan,
               modal_feature=Counter(chosen).most_common(1)[0][0],
               threshold_pctl_cv=round(float(np.std(thp) / np.mean(thp)), 3) if np.mean(thp) else np.nan)
    if return_oof:
        oof = dict(score=np.concatenate(oof_s), y=np.concatenate(oof_y),
                   year=np.concatenate(oof_yr), wn=np.concatenate(oof_wn))
        return res, oof
    return res


def _unit_row(scale, unit, g, cc, rc):
    g = g.sort_values(["YR", "WN"]).copy()
    yr = g["YR"].to_numpy()
    g["_ta"] = _continuous_ta(g[cc].to_numpy(float)) if TA_MODE == "continuous" else _constant_ta(g[cc].to_numpy(float), yr)
    feats = _build_features(g, rc)
    tg = {L: _onset_target(g, L) for L in EFF_LEADS}
    full = np.ones(len(g), dtype=bool)
    accum = {k: v for k, v in feats.items() if "accum" in k}
    dyn = {k: v for k, v in feats.items() if ("rate" in k or "stalta" in k)}

    # ---- headline: interpretable accumulation threshold in mm (rain >= T) ----
    ca = _select(accum, tg, full)
    if not ca:
        return None
    fa, La, tha = ca["feature"], ca["lead"], ca["threshold"]
    xa = feats[fa]; pctl = float(np.nanmean(xa < tha) * 100)
    ml = _median_lead(g, xa, tha, ">=")
    la, accum_oof = _loyo(feats, accum, tg, yr, wn=g["WN"].to_numpy(), return_oof=True)

    # ---- secondary: best rainfall-dynamics representation (documents the lift) ----
    cd = _select(dyn, tg, full)
    if cd:
        ld, dyn_oof = _loyo(feats, dyn, tg, yr, wn=g["WN"].to_numpy(), return_oof=True)
    else:
        ld, dyn_oof = {}, dict(score=np.array([]), y=np.array([]), year=np.array([]), wn=np.array([]))
    dyn_ml = _median_lead(g, feats[cd["feature"]], cd["threshold"], ">=") if cd else np.nan

    # per-unit onset-AUROC CI vs chance (cluster bootstrap over years) for the headline
    a_lo = a_hi = np.nan; beats = False
    try:
        import stats as _st
        if len(accum_oof["y"]) >= 5 and len(np.unique(accum_oof["y"])) > 1:
            r = _st.auroc_ci_vs_chance(accum_oof["y"], accum_oof["score"], accum_oof["year"], B=_BOOT_B)
            a_lo, a_hi, beats = r["ci"][0], r["ci"][1], r["beats_chance"]
    except Exception:
        pass

    row = dict(scale=scale, unit=unit, ta_mode=TA_MODE, feature=fa,
               target="TA_onset_within_lead", lead=La, direction=">=",
               threshold_mm=round(tha, 1), threshold_pctl=round(pctl, 1),
               auroc_insample=round(ca["auroc"], 3), youden_j=round(ca["youden_j"], 3),
               sensitivity=round(ca["sensitivity"], 3), specificity=round(ca["specificity"], 3),
               auroc_loyo=la["auroc_loyo"], balanced_acc_loyo=la["balanced_acc_loyo"],
               auroc_loyo_ci_lo=round(a_lo, 3) if a_lo == a_lo else np.nan,
               auroc_loyo_ci_hi=round(a_hi, 3) if a_hi == a_hi else np.nan,
               beats_chance=bool(beats),
               sens_loyo=la["sens_loyo"], spec_loyo=la["spec_loyo"],
               accum_median_lead_weeks=round(ml, 1) if ml == ml else np.nan,
               modal_feature=la["modal_feature"], threshold_pctl_cv=la["threshold_pctl_cv"],
               dyn_feature=cd["feature"] if cd else None,
               dyn_lead=cd["lead"] if cd else np.nan,
               dyn_threshold=round(cd["threshold"], 3) if cd else np.nan,
               dyn_auroc_insample=round(cd["auroc"], 3) if cd else np.nan,
               dyn_sensitivity=round(cd["sensitivity"], 3) if cd else np.nan,
               dyn_specificity=round(cd["specificity"], 3) if cd else np.nan,
               dyn_auroc_loyo=ld.get("auroc_loyo", np.nan) if cd else np.nan,
               dyn_median_lead_weeks=round(dyn_ml, 1) if dyn_ml == dyn_ml else np.nan,
               n=ca["n"])
    return row, accum_oof, dyn_oof


def derive(scale, enr=None, boot_B=2000):
    global _BOOT_B
    _BOOT_B = int(boot_B)
    if enr is None:
        enr = pd.read_csv(os.path.join(ROOT, "outputs", scale, "enriched.csv"))
    cc = _col(enr.columns, "DC_"); rc = _col(enr.columns, "RF_")
    if cc is None or rc is None:
        print("  rain_threshold: case/rainfall column missing; skipping"); return None, None
    uc = _unit_col(enr.columns)
    lab = enr[enr.get("labelable", 1) == 1].copy()
    units = list(lab.groupby(uc)) if uc else [("(pooled)", lab)]
    rows = []; acc_pool = {}; dyn_pool = {}
    for u, g in units:
        try:
            res = _unit_row(scale, u, g, cc, rc)
            if res:
                r, a_oof, d_oof = res
                rows.append(r)
                # key each OOF row by (unit, year, wn) so accum & dyn can be paired
                for arr, pool in ((a_oof, acc_pool), (d_oof, dyn_pool)):
                    for sc, yy, yr_, wn_ in zip(arr["score"], arr["y"], arr["year"], arr["wn"]):
                        pool[(str(u), str(yr_), int(wn_))] = (float(sc), int(yy))
        except Exception as e:
            print(f"    {u}: skipped ({e})")
    out = pd.DataFrame(rows)
    if uc and len(rows) > 1:
        num = out.select_dtypes("number").median(numeric_only=True)
        med = dict(scale=scale, unit="(scale median)", ta_mode=TA_MODE, target="TA_onset_within_lead")
        for k in out.columns:
            if k in num:
                med[k] = round(float(num[k]), 3)
        med["modal_feature"] = Counter(out["modal_feature"].dropna()).most_common(1)[0][0]
        med["feature"] = Counter(out["feature"].dropna()).most_common(1)[0][0]
        dynf = out["dyn_feature"].dropna()
        med["dyn_feature"] = Counter(dynf).most_common(1)[0][0] if len(dynf) else None
        out = pd.concat([pd.DataFrame([med]), out], ignore_index=True)
    path = os.path.join(ROOT, "outputs", scale, "rain_thresholds.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)

    # Delphi revision: paired dynamics-vs-accumulation onset-AUROC lift, clustered by
    # unit-year, on the weeks both representations score. This is the inference behind
    # the README claim that rainfall DYNAMICS beats a raw accumulation LEVEL.
    rain_stats = None
    try:
        import stats as _st
        common = [k for k in acc_pool if k in dyn_pool]
        if len(common) >= 30:
            y = np.array([acc_pool[k][1] for k in common])
            sa = np.array([acc_pool[k][0] for k in common])
            sd = np.array([dyn_pool[k][0] for k in common])
            cl = np.array([f"{k[0]}-{k[1]}" for k in common])
            pcb = _st.paired_cluster_bootstrap(
                y, {"accumulation": sa, "dynamics": sd}, cl, B=_BOOT_B, metric_names=["auroc"])
            dd = pcb["auroc"]["delta"]; key = list(dd)[0]  # accumulation_vs_dynamics = accum - dyn
            d_ad = dd[key]
            dyn_minus_accum = dict(delta=-d_ad["delta"],
                                   ci=[-d_ad["ci"][1], -d_ad["ci"][0]], p=d_ad["p"])
            n_bad = int((~out.get("beats_chance", pd.Series([], dtype=bool)).fillna(False)).sum())
            rain_stats = dict(target="TA_onset_within_lead", aligned_on="(unit-year, week)",
                              n_common=int(len(common)), n_clusters=int(len(np.unique(cl))),
                              accumulation_auroc=pcb["auroc"]["point"]["accumulation"],
                              dynamics_auroc=pcb["auroc"]["point"]["dynamics"],
                              dyn_minus_accum=dyn_minus_accum,
                              n_units_not_clearing_0p5=n_bad)
    except Exception as e:
        print(f"  rain_threshold stats skipped: {e}")
    msg = f"  wrote rain_thresholds.csv ({len(out)} rows; accum-mm headline + dynamics secondary, LOYO)"
    if rain_stats:
        dm = rain_stats["dyn_minus_accum"]
        msg += (f"; dyn−accum AUROC {dm['delta']:+.3f} "
                f"CI[{dm['ci'][0]:+.3f},{dm['ci'][1]:+.3f}] p={dm['p']:.3f}")
    print(msg)
    return out, rain_stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="qc")
    derive(ap.parse_args().scale)
