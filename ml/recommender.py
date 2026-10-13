"""
AgriSmart recommendation engine  (used by the website and the experiments)
=========================================================================

NOVEL CONTRIBUTIONS (each one is evaluated or demonstrated in ml/experiments.py)

N1  Quality-Weighted Temporal Consensus (QW-TC)
    Conventional crop recommenders classify ONE soil snapshot.  AgriSmart keeps
    a time series of readings per plot in MySQL, so it builds a consensus
    feature vector from every reading in a recent window (default 7 days):

        x_consensus = sum_i q_i * w_i * x_i / sum_i q_i * w_i

    q_i = data-quality score of reading i (range check, robust spike test,
          cross-sensor consistency, known sensor fault codes such as the
          DS18B20's -127 / 85 degC values)
    w_i = day-blocked recency weight exp(-age_days*24 / tau), tau = 7 days
    and classifies that.  (Exp 3 showed that averaging features beats
    averaging per-reading probabilities, because single readings carry the
    day/night swing while crop requirements describe daily-mean conditions.)

N2  Feature-availability adaptive model selection
    pH is optional on our hardware.  If a valid pH value exists for the plot
    (sensor or manual, within PH_VALID_DAYS) the "with_pH" model is used,
    otherwise the "without_pH" model - the system never refuses to answer just
    because a cheap sensor is missing.

N3  Confidence-aware abstention
    An Isolation-Forest out-of-distribution detector and a probability
    threshold mark a recommendation as LOW_CONFIDENCE or OUT_OF_DISTRIBUTION,
    and the UI asks for a lab soil test instead of presenting a guess as fact.

N4  Context-aware re-ranking with a Virtual Nitrogen Balance Index (VNBI)
    Because there is no NPK sensor, the plot's nitrogen status is ESTIMATED
    from its own records in the database: the N-demand of the last three crops
    grown and the nitrogen fertiliser logged in Treatments.  The ML ranking is
    then re-ranked by
        season fit  x  rotation fit (same crop blocked, same family penalised,
                       legume after cereal rewarded)  x  nutrient fit (VNBI)

N5  Explainable "fit card"
    For each recommended crop the engine shows which measured value lies
    inside / below / above the crop's learned P5-P95 envelope (learned from
    the training data, stored in the MySQL crop catalogue).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .common import FEATURES_FULL, FEATURES_NO_PH, MODEL_DIR, SENSOR_RANGES
    from . import crop_knowledge as ck
except ImportError:  # executed as a script from ml/
    from common import FEATURES_FULL, FEATURES_NO_PH, MODEL_DIR, SENSOR_RANGES
    import crop_knowledge as ck

NUM_SENSOR = ["moisture_pct", "soil_temp_c", "air_temp_c", "humidity_pct"]
FAULT_CODES = {"soil_temp_c": {-127.0, 85.0}}   # DS18B20 disconnected / power-on reset
TAU_HOURS = 168.0          # 7-day decay (Exp 3 ablation: hourly tau=72 h hurt accuracy)
PH_VALID_DAYS = 180
LOW_CONF_THRESHOLD = 0.50  # chosen from the Exp 5 risk-coverage curve
N_DEMAND_SCORE = {"FIXER": +1.0, "LOW": 0.0, "MEDIUM": -0.5, "HIGH": -1.0}


def fill_missing_soil_temp(readings: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """A node with no DS18B20 sends no soil temperature at all. Estimate it with the
    noise-free formula that created soil_temp_c in the training data (ml/build_dataset.py):
        soil_temp = air_temp + 1.5 - 0.08 * (moisture - 20)
    Only used when NO reading in the window has a soil temperature, so a real probe
    that faults (-127 / 85 -> NULL) is still treated as invalid by quality_scores()."""
    df = readings.copy()
    if len(df) == 0 or "soil_temp_c" not in df or df["soil_temp_c"].notna().any():
        return df, False
    est = df["air_temp_c"].astype(float) + 1.5 - 0.08 * (df["moisture_pct"].astype(float) - 20)
    df["soil_temp_c"] = est.round(2)
    return df, bool(est.notna().any())


# --------------------------------------------------------------------------- N1
def quality_scores(readings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `readings` with a `quality` column in [0,1] and a
    human-readable `flags` column."""
    df = readings.copy().reset_index(drop=True)
    q = np.ones(len(df))
    flags = [[] for _ in range(len(df))]

    for col in NUM_SENSOR:
        if col not in df:
            continue
        v = df[col].astype(float)
        lo, hi = SENSOR_RANGES[col]
        # 1) fault codes / missing / outside physical range -> unusable
        bad = v.isna() | (v < lo) | (v > hi) | v.isin(FAULT_CODES.get(col, set()))
        for i in np.where(bad)[0]:
            q[i] = 0.0; flags[i].append(f"{col}:invalid")
        # 2) robust spike test against the window median (MAD z-score)
        good = v[~bad]
        if len(good) >= 5:
            med = good.median()
            mad = (good - med).abs().median() * 1.4826 + 1e-6
            floor = {"moisture_pct": 1.5, "soil_temp_c": 1.0, "air_temp_c": 2.0, "humidity_pct": 5.0}[col]
            z = (v - med).abs() / max(mad, floor)
            for i in np.where((~bad) & (z > 3.5))[0]:
                q[i] *= max(0.0, 1 - (z[i] - 3.5) / 3.5); flags[i].append(f"{col}:spike")
    # 3) probe out of soil: capacitive sensor reads ~0 % in air
    if "moisture_pct" in df:
        for i in np.where(df["moisture_pct"].astype(float) < 1.0)[0]:
            q[i] *= 0.1; flags[i].append("moisture:probe_dry/out_of_soil")
    # 4) cross-sensor consistency: soil and air temperature cannot differ wildly
    if {"soil_temp_c", "air_temp_c"} <= set(df):
        d = (df["soil_temp_c"] - df["air_temp_c"]).abs()
        for i in np.where(d > 15)[0]:
            q[i] *= 0.5; flags[i].append("temp:inconsistent")

    df["quality"] = np.round(q, 3)
    df["flags"] = [",".join(f) for f in flags]
    return df


def recency_weights(ts: pd.Series, now: datetime | None = None, tau: float = TAU_HOURS) -> np.ndarray:
    """Day-blocked recency weight: every reading of the same day gets the same
    weight, so the day/night balance of the samples is preserved (an hourly
    decay over-weights the most recent hours and biases the mean)."""
    ts = pd.to_datetime(ts)
    now = pd.Timestamp(now) if now is not None else ts.max()
    age_days = ((now - ts).dt.total_seconds().clip(lower=0) // 86400).to_numpy()
    return np.exp(-age_days * 24 / tau)


# --------------------------------------------------------------------------- N4
def virtual_nitrogen_index(history: list[dict]) -> float:
    """history: most-recent-first list of {"crop": str, "n_applied_kg_acre": float}
    for the plot's last cycles.  Returns VNBI roughly in [-2, +2]:
    negative = soil likely depleted of N, positive = N-rich (legumes / fertiliser)."""
    vnbi, w = 0.0, 1.0
    for h in history[:3]:
        crop = h.get("crop")
        if crop in ck.CROPS:
            vnbi += w * N_DEMAND_SCORE[ck.info(crop)["n_demand"]]
        vnbi += w * min(1.0, float(h.get("n_applied_kg_acre") or 0) / 40.0)
        w *= 0.6
    return round(vnbi, 3)


def context_factors(crop: str, month: int, last_crop: str | None, vnbi: float):
    info = ck.info(crop)
    reasons = []
    # season
    if info["is_perennial"] or info["season"] == ck.season_for_month(month):
        s = 1.0
    else:
        s = 0.6; reasons.append(f"off-season (best sown in {info['season']})")
    # rotation
    r = 1.0
    if last_crop and last_crop in ck.CROPS:
        last = ck.info(last_crop)
        if last_crop == crop and not info["is_perennial"]:
            r = 0.0; reasons.append("same crop as last cycle - blocked by rotation rule")
        elif last["family"] == info["family"] and not info["is_perennial"]:
            r = 0.8; reasons.append(f"same family ({info['family']}) as last crop")
        elif last["family"] in ("Cereal", "Fibre") and info["family"] == "Legume":
            r = 1.15; reasons.append("legume after cereal/fibre restores soil N")
    # nutrient (VNBI)
    n = 1.0
    if vnbi < -0.75:
        if info["n_demand"] == "HIGH":
            n = 0.85; reasons.append("soil N likely depleted (VNBI) - high-N crop penalised")
        elif info["n_demand"] == "FIXER":
            n = 1.15; reasons.append("soil N likely depleted (VNBI) - N-fixing legume favoured")
    elif vnbi > 0.75 and info["n_demand"] == "HIGH":
        n = 1.1; reasons.append("soil N likely high (VNBI) - suits high-N crop")
    return s, r, n, reasons


# --------------------------------------------------------------------------- engine
@dataclass
class Recommendation:
    status: str
    model_used: str
    n_readings: int
    n_readings_used: int
    mean_quality: float
    consensus_features: dict
    vnbi: float | None
    season_now: str
    ranking: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__


class AgriRecommender:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.models = {t: joblib.load(model_dir / f"crop_model_{t}.joblib") for t in ("with_pH", "without_pH")}
        self.ood = {t: joblib.load(model_dir / f"ood_detector_{t}.joblib") for t in ("with_pH", "without_pH")}
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.env = self.meta["envelopes"]

    # ---- low level -------------------------------------------------------
    def _features(self, tag):
        return FEATURES_FULL if tag == "with_pH" else FEATURES_NO_PH

    def predict_proba(self, X: pd.DataFrame, tag: str) -> np.ndarray:
        return self.models[tag].predict_proba(X[self._features(tag)])

    @property
    def classes(self):
        return list(self.models["with_pH"].classes_)

    def recommend_features(self, feats: dict, soil_type: str, ph: float | None = None, **kw):
        """Convenience: recommend from ONE manual reading (e.g. a form entry)."""
        df = pd.DataFrame([{"log_ts": pd.Timestamp.now(), **{c: feats[c] for c in NUM_SENSOR}}])
        return self.recommend(df, soil_type, ph, **kw)

    def fit_card(self, crop: str, feats: dict, soil_type: str) -> list:
        card = []
        for f, (lo, hi) in ((k, v) for k, v in self.env[crop].items() if k != "soil_types"):
            val = feats.get(f)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            state = "ok" if lo <= val <= hi else ("low" if val < lo else "high")
            card.append({"feature": f, "value": round(float(val), 2), "lo": lo, "hi": hi, "state": state})
        share = self.env[crop]["soil_types"].get(soil_type, 0.0)
        card.append({"feature": "soil_type", "value": soil_type, "lo": None, "hi": None,
                     "state": "ok" if share >= 0.15 else ("low" if share > 0 else "high"),
                     "share": share})
        return card

    # ---- main entry point -----------------------------------------------
    def recommend(self, readings: pd.DataFrame, soil_type: str, ph: float | None = None,
                  history: list[dict] | None = None, month: int | None = None,
                  now: datetime | None = None, top_k: int = 3, use_consensus: bool = True,
                  use_context: bool = True) -> Recommendation:
        """readings: DataFrame with columns log_ts + NUM_SENSOR (one row per reading)."""
        month = month or (now or datetime.now()).month
        history = history or []
        msgs = []
        readings = readings.sort_values("log_ts").reset_index(drop=True)
        if not use_consensus:
            readings = readings.tail(1)

        tag = "with_pH" if ph is not None and SENSOR_RANGES["ph"][0] <= ph <= SENSOR_RANGES["ph"][1] else "without_pH"
        msgs.append("pH available - using with-pH model" if tag == "with_pH"
                    else "No valid pH - using sensor-only model (add a pH test for higher accuracy)")

        readings, estimated = fill_missing_soil_temp(readings)
        if estimated:
            msgs.append("No soil-temperature probe - soil temperature estimated from air temperature and moisture")
        qdf = quality_scores(readings)
        w = qdf["quality"].to_numpy() * recency_weights(qdf["log_ts"], now)
        used = qdf[qdf["quality"] > 0]
        if len(qdf) == 0 or w.sum() <= 1e-9:
            return Recommendation("NO_VALID_DATA", tag, len(qdf), 0, 0.0, {}, None,
                                  ck.season_for_month(month), [], msgs + ["All readings failed quality checks"])

        X = qdf[NUM_SENSOR].copy()
        X["ph"] = ph if ph is not None else np.nan
        X["soil_type"] = soil_type
        valid = w > 0
        # consensus feature vector = quality x recency weighted mean of good readings
        cf = {c: float(np.average(X.loc[valid, c], weights=w[valid])) for c in NUM_SENSOR}
        cf["ph"] = ph
        Xc = pd.DataFrame([{**cf, "ph": ph if ph is not None else np.nan, "soil_type": soil_type}])
        p = self.predict_proba(Xc, tag)[0]
        is_ood = self.ood[tag].predict(Xc[self._features(tag)])[0] == -1

        last_crop = history[0]["crop"] if history else None
        vnbi = virtual_nitrogen_index(history) if history else None
        rows = []
        for crop, pm in zip(self.classes, p):
            s, r, n, why = (context_factors(crop, month, last_crop, vnbi or 0.0) if use_context
                            else (1.0, 1.0, 1.0, []))
            rows.append({"crop": crop, "p_ml": float(pm), "season_f": s, "rotation_f": r,
                         "nutrient_f": n, "raw": float(pm) * s * r * n, "context_notes": why})
        for x in rows:   # NOT re-normalised: a blocked crop must not inflate weak ones
            x["score"] = min(1.0, x["raw"])
        rows.sort(key=lambda x: x["score"], reverse=True)
        top = rows[:top_k]
        for x in top:
            x["info"] = ck.info(x["crop"])
            x["fit_card"] = self.fit_card(x["crop"], cf, soil_type)

        status = "CONFIDENT"
        if is_ood:
            status = "OUT_OF_DISTRIBUTION"
            msgs.append("Readings are unlike anything in the training data - verify sensors / get a lab soil test.")
        elif top[0]["p_ml"] < LOW_CONF_THRESHOLD:
            status = "LOW_CONFIDENCE"
            msgs.append("Model is unsure between several crops - treat the list as options, not a verdict.")
        bad = (qdf["quality"] < 1).sum()
        if bad:
            msgs.append(f"{bad} of {len(qdf)} readings were down-weighted or rejected by quality checks.")

        return Recommendation(status, tag, len(qdf), int(len(used)), round(float(qdf["quality"].mean()), 3),
                              {k: (round(v, 2) if v is not None else None) for k, v in cf.items()},
                              vnbi, ck.season_for_month(month), top, msgs)
