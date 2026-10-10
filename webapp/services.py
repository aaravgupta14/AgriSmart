"""Business logic that connects MySQL and the ML recommendation engine."""
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
from webapp.db import execute, get_db, query

sys.path.insert(0, str(config.ROOT / "ml"))
from recommender import NUM_SENSOR, AgriRecommender, quality_scores  # noqa: E402

_engine = None


def engine() -> AgriRecommender:
    global _engine
    if _engine is None:
        _engine = AgriRecommender()
    return _engine


def plot_window(plot_id: int, days: int = config.RECOMMEND_WINDOW_DAYS) -> pd.DataFrame:
    rows = query(f"""
        SELECT log_id, log_ts, {", ".join(NUM_SENSOR)}
        FROM soil_health_logs
        WHERE plot_id = %s
          AND (moisture_pct IS NOT NULL OR air_temp_c IS NOT NULL)
          AND log_ts >= (SELECT MAX(log_ts) FROM soil_health_logs WHERE plot_id = %s) - INTERVAL %s DAY
        ORDER BY log_ts""", (plot_id, plot_id, days))
    df = pd.DataFrame(rows, columns=["log_id", "log_ts", *NUM_SENSOR])
    for c in NUM_SENSOR:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # NULL soil temp (fault code removed by trigger) -> treat as invalid reading
    return df


def plot_history(plot_id: int) -> list[dict]:
    """Most-recent-first list of {crop, n_applied_kg_acre} for the VNBI."""
    rows = query("""
        SELECT c.crop_name AS crop,
               COALESCE((SELECT SUM(t.n_kg) FROM treatments t WHERE t.cycle_id = cc.cycle_id), 0) / p.area_acres
                   AS n_applied_kg_acre
        FROM cultivation_cycles cc JOIN crops c ON c.crop_id = cc.crop_id JOIN plots p ON p.plot_id = cc.plot_id
        WHERE cc.plot_id = %s ORDER BY cc.sow_date DESC, cc.cycle_id DESC LIMIT 3""", (plot_id,))
    return [{"crop": r["crop"], "n_applied_kg_acre": float(r["n_applied_kg_acre"])} for r in rows]


def recommend_for_plot(plot_id: int, month: int | None = None, save: bool = True) -> dict:
    plot = query("SELECT * FROM plots WHERE plot_id = %s", (plot_id,), one=True)
    if not plot:
        raise ValueError("Plot not found")
    win = plot_window(plot_id)
    ph_row = query("SELECT ph_value, ph_measured_on, ph_source FROM v_latest_ph WHERE plot_id = %s", (plot_id,), one=True)
    ph = float(ph_row["ph_value"]) if ph_row else None
    hist = plot_history(plot_id)
    rec = engine().recommend(win, plot["soil_type"], ph, hist, month=month or datetime.now().month)
    result = rec.to_dict()
    result.update(plot=plot, ph_info=ph_row, history=hist,
                  window_start=win.log_ts.min() if len(win) else None,
                  window_end=win.log_ts.max() if len(win) else None)

    if len(win):  # store per-reading quality so SQL views can exclude rejected rows
        q = quality_scores(win)
        with get_db().cursor() as cur:
            cur.executemany("UPDATE soil_health_logs SET quality_score = %s WHERE log_id = %s",
                            [(float(a), int(b)) for a, b in zip(q.quality, q.log_id)])
        result["quality_table"] = q.tail(48).iloc[::-1].to_dict("records")

    if save and rec.ranking:
        run_id = execute("""INSERT INTO recommendation_runs
            (plot_id, model_used, status, window_start, window_end, n_readings, n_readings_used,
             mean_quality, vnbi, season_at_run) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (plot_id, rec.model_used, rec.status, result["window_start"], result["window_end"],
             rec.n_readings, rec.n_readings_used, rec.mean_quality, rec.vnbi, rec.season_now))
        ids = {r["crop_name"]: r["crop_id"] for r in query("SELECT crop_id, crop_name FROM crops")}
        with get_db().cursor() as cur:
            cur.executemany("""INSERT INTO recommendation_items
                (run_id, rank_no, crop_id, p_ml, season_f, rotation_f, nutrient_f, final_score)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(run_id, i + 1, ids[x["crop"]], x["p_ml"], x["season_f"], x["rotation_f"],
                  x["nutrient_f"], x["score"]) for i, x in enumerate(rec.ranking)])
        result["run_id"] = run_id
    return result


def recommend_manual(values: dict, soil_type: str, ph, last_crop: str | None, month: int) -> dict:
    hist = [{"crop": last_crop, "n_applied_kg_acre": 0}] if last_crop else []
    rec = engine().recommend_features(values, soil_type, ph, history=hist, month=month)
    return rec.to_dict()


def validate_reading(d: dict) -> tuple[dict, list]:
    """Same range checks the serial bridge applies (edge validation)."""
    from common import SENSOR_RANGES
    clean, errors = {}, []
    for k in NUM_SENSOR + ["ph"]:
        v = d.get(k if k != "ph" else "ph_value", d.get(k))
        if v in (None, "", "nan", "NaN"):
            clean[k] = None; continue
        try:
            v = float(v)
        except ValueError:
            errors.append(f"{k}: not a number"); continue
        lo, hi = SENSOR_RANGES[k]
        if k == "soil_temp_c" and v in (-127.0, 85.0):
            clean[k] = v  # trigger will null it and annotate the row
        elif not lo <= v <= hi:
            errors.append(f"{k}={v} outside {lo}-{hi}")
        else:
            clean[k] = v
    if all(clean.get(k) is None for k in NUM_SENSOR + ["ph"]):
        errors.append("no values supplied")
    return clean, errors


def nan_to_none(o):
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [nan_to_none(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat(sep=" ", timespec="minutes")
    return o
