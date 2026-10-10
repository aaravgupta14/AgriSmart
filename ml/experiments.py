"""
AgriSmart - Novelty experiments (run AFTER train_models.py)
===========================================================
exp3  Quality-Weighted Temporal Consensus (QW-TC) vs snapshot classification
      under realistic sensor faults  (novelty N1)
exp4  Feature-availability adaptive model selection when pH is missing (N2)
exp5  Confidence-aware abstention: risk-coverage + out-of-distribution test (N3)
exp6  Context-aware re-ranking with the Virtual Nitrogen Balance Index -
      scenario case study (N4)
Finally writes experiments/EXPERIMENTS_SUMMARY.md with every table.

All models here are trained on the 80 % TRAIN split only and evaluated on the
20 % TEST split (same split/seed as train_models.py) - no leakage.

SIMULATION ASSUMPTIONS (state these in the report)
 * Each dataset row is taken as the plot's DAILY-MEAN condition.
 * A 7-day window with one reading every 4 h (42 readings) is generated around
   it with diurnal cycles (air temp +/-4 degC, RH -/+8 %, soil temp +/-1.5 degC),
   slow moisture drift, and sensor noise of the order of the datasheet accuracy
   (DHT11 +/-2 degC, +/-5 %RH; DS18B20 +/-0.5 degC; capacitive moisture ~1 %).
 * Faults injected with probability f per reading, chosen uniformly from:
   DS18B20 disconnected (-127 degC), DS18B20 power-on reset (85 degC),
   moisture probe out of soil (0 %), humidity spike (+/-30 %RH),
   moisture spike (+/-15 %).
Run:  python ml/experiments.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import crop_knowledge as ck
from common import DATA, FEATURES_FULL, FEATURES_NO_PH, MODEL_DIR, ROOT, SENSOR_RANGES
from recommender import (NUM_SENSOR, AgriRecommender, context_factors, quality_scores,
                         recency_weights, virtual_nitrogen_index)

SEED = 42
EXP = ROOT / "experiments"
FAULT_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
N_READ, STEP_H = 42, 4


def split():
    df = pd.read_csv(DATA)
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def fit_split_models(tr):
    """Clone the deployed pipelines and refit them on the train split only."""
    out = {}
    for tag, feats in (("with_pH", FEATURES_FULL), ("without_pH", FEATURES_NO_PH)):
        m = clone(__import__("joblib").load(MODEL_DIR / f"crop_model_{tag}.joblib")).fit(tr[feats], tr.label)
        out[tag] = m
    return out


def topk_hit(proba_row, classes, truth, k=3):
    return truth in classes[np.argsort(proba_row)[::-1][:k]]


# ----------------------------------------------------------------------------- exp3
def simulate_window(row, f, rng):
    h = np.arange(N_READ) * STEP_H + rng.integers(0, 24)
    ts = pd.Timestamp("2026-09-01") + pd.to_timedelta(np.arange(N_READ) * STEP_H, unit="h")
    ang_air = 2 * np.pi * (h - 9) / 24
    air = row.air_temp_c + 4 * np.sin(ang_air) + rng.normal(0, 1.0, N_READ)
    hum = np.clip(row.humidity_pct - 8 * np.sin(ang_air) + rng.normal(0, 2.5, N_READ), 5, 100)
    soil = row.soil_temp_c + 1.5 * np.sin(2 * np.pi * (h - 12) / 24) + rng.normal(0, 0.25, N_READ)
    moist = row.moisture_pct + np.cumsum(rng.normal(0, 0.15, N_READ)) + rng.normal(0, 1.0, N_READ)
    w = pd.DataFrame({"log_ts": ts, "moisture_pct": np.round(np.clip(moist, 0, 60), 1),
                      "soil_temp_c": np.round(soil / 0.0625) * 0.0625,
                      "air_temp_c": np.round(air), "humidity_pct": np.round(hum)})
    faults = rng.random(N_READ) < f
    for i in np.where(faults)[0]:
        kind = rng.integers(5)
        if kind == 0: w.loc[i, "soil_temp_c"] = -127.0
        elif kind == 1: w.loc[i, "soil_temp_c"] = 85.0
        elif kind == 2: w.loc[i, "moisture_pct"] = 0.0
        elif kind == 3: w.loc[i, "humidity_pct"] = float(np.clip(w.loc[i, "humidity_pct"] + rng.choice([-30, 30]), 0, 100))
        else: w.loc[i, "moisture_pct"] = float(max(0, w.loc[i, "moisture_pct"] + rng.choice([-15, 15])))
    return w


def range_valid(w):
    ok = np.ones(len(w), bool)
    for c in NUM_SENSOR:
        lo, hi = SENSOR_RANGES[c]
        ok &= (w[c] >= lo) & (w[c] <= hi)
    ok &= ~w.soil_temp_c.isin([-127.0, 85.0])
    return ok.to_numpy()


def exp3(models, te):
    out = EXP / "exp3_temporal_consensus"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tag, feats = "with_pH", FEATURES_FULL
    m = models[tag]; classes = m.classes_
    methods = ["A_snapshot_raw", "B_snapshot_validated", "C_feature_mean_raw",
               "D_prob_mean_validated", "E_feature_mean_validated", "F_QW-TC_no_recency (ablation)",
               "G_QW-TC_hourly_recency_tau72h (ablation)", "H_QW-TC (proposed)"]
    rows = []
    for f in FAULT_RATES:
        # 1) simulate every window, collect inputs; 2) predict in two batches (fast)
        wins, mean_rows, idx = [], {k: [] for k in ("C", "E", "F", "G", "H")}, []
        for _, r in te.iterrows():
            w = simulate_window(r, f, rng)
            X = w[NUM_SENSOR].copy(); X["ph"] = r.ph; X["soil_type"] = r.soil_type
            ok = range_valid(w)
            q = quality_scores(w)["quality"].to_numpy()
            wt = q * recency_weights(w.log_ts)
            age_h = (w.log_ts.max() - w.log_ts).dt.total_seconds().to_numpy() / 3600
            wt_hourly = q * np.exp(-age_h / 72)
            fm = lambda weights: {**{c: float(np.average(X[c], weights=weights)) for c in NUM_SENSOR},
                                  "ph": r.ph, "soil_type": r.soil_type}
            mean_rows["C"].append(fm(np.ones(len(w))))
            mean_rows["E"].append(fm(ok.astype(float)) if ok.any() else X.iloc[-1].to_dict())
            mean_rows["F"].append(fm(q) if q.sum() > 0 else X.iloc[-1].to_dict())
            mean_rows["G"].append(fm(wt_hourly) if wt_hourly.sum() > 0 else X.iloc[-1].to_dict())
            mean_rows["H"].append(fm(wt) if wt.sum() > 0 else X.iloc[-1].to_dict())
            wins.append((X, ok)); idx.append(len(X))
        P_all = m.predict_proba(pd.concat([x for x, _ in wins], ignore_index=True)[feats])
        P_mean = {k: m.predict_proba(pd.DataFrame(v)[feats]) for k, v in mean_rows.items()}
        hits = {k: [] for k in methods}; top3 = {k: [] for k in methods}
        start = 0
        for n_i, ((X, ok), truth) in enumerate(zip(wins, te.label)):
            P = P_all[start:start + len(X)]; start += len(X)
            last_ok = np.where(ok)[0][-1] if ok.any() else len(X) - 1
            preds = {"A_snapshot_raw": P[-1], "B_snapshot_validated": P[last_ok],
                     "C_feature_mean_raw": P_mean["C"][n_i],
                     "D_prob_mean_validated": P[ok].mean(0) if ok.any() else P.mean(0),
                     "E_feature_mean_validated": P_mean["E"][n_i],
                     "F_QW-TC_no_recency (ablation)": P_mean["F"][n_i],
                     "G_QW-TC_hourly_recency_tau72h (ablation)": P_mean["G"][n_i],
                     "H_QW-TC (proposed)": P_mean["H"][n_i]}
            for k, p in preds.items():
                hits[k].append(classes[p.argmax()] == truth)
                top3[k].append(topk_hit(p, classes, truth))
        for k in methods:
            rows.append({"fault_rate": f, "method": k, "top1_accuracy": np.mean(hits[k]),
                         "top3_accuracy": np.mean(top3[k])})
        print(f"  f={f:.2f} " + " | ".join(f"{k.split('_')[0]}={np.mean(hits[k]):.3f}" for k in methods), flush=True)
    res = pd.DataFrame(rows); res.round(4).to_csv(out / "results.csv", index=False)
    piv = res.pivot(index="fault_rate", columns="method", values="top1_accuracy")
    piv.round(4).to_csv(out / "results_pivot_top1.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for c in methods:
        ax.plot(piv.index * 100, piv[c], marker="o", lw=2.5 if "proposed" in c else 1.2, label=c)
    ax.set_xlabel("fault rate per reading (%)"); ax.set_ylabel("top-1 accuracy (test split)")
    ax.set_title("Exp 3 - robustness to sensor faults (7-day window, 42 readings)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5)); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "chart.png", dpi=150)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- exp4
def exp4(models, tr, te):
    out = EXP / "exp4_adaptive_model_selection"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    full = models["with_pH"]; nph = models["without_pH"]
    rows.append({"scenario": "pH measured", "strategy": "with-pH model",
                 "accuracy": accuracy_score(te.label, full.predict(te[FEATURES_FULL]))})
    for name, fill in [("training median", tr.ph.median()), ("neutral 7.0", 7.0)]:
        X = te[FEATURES_FULL].copy(); X["ph"] = fill
        rows.append({"scenario": "pH missing", "strategy": f"with-pH model, pH imputed = {name}",
                     "accuracy": accuracy_score(te.label, full.predict(X))})
    rows.append({"scenario": "pH missing", "strategy": "sensor-only model (proposed adaptive switch)",
                 "accuracy": accuracy_score(te.label, nph.predict(te[FEATURES_NO_PH]))})
    res = pd.DataFrame(rows); res.round(4).to_csv(out / "results.csv", index=False)
    print(res.round(4).to_string(index=False))
    return res


# ----------------------------------------------------------------------------- exp5
def exp5(models, tr, te):
    out = EXP / "exp5_abstention"; out.mkdir(parents=True, exist_ok=True)
    m = models["with_pH"]; classes = m.classes_
    P = m.predict_proba(te[FEATURES_FULL]); conf = P.max(1)
    correct = classes[P.argmax(1)] == te.label.to_numpy()
    rc = []
    for t in np.round(np.arange(0.0, 0.91, 0.05), 2):
        acc_mask = conf >= t
        rc.append({"threshold": t, "coverage": acc_mask.mean(),
                   "accuracy_on_accepted": correct[acc_mask].mean() if acc_mask.any() else np.nan,
                   "accuracy_on_flagged": correct[~acc_mask].mean() if (~acc_mask).any() else np.nan})
    rc = pd.DataFrame(rc); rc.round(4).to_csv(out / "risk_coverage.csv", index=False)

    from sklearn.pipeline import Pipeline
    from train_models import make_pre
    iso = Pipeline([("pre", make_pre(FEATURES_FULL)),
                    ("iso", IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED))]).fit(tr[FEATURES_FULL])
    rng = np.random.default_rng(SEED)
    n = 1000
    junk = pd.DataFrame({c: rng.uniform(*SENSOR_RANGES[c], n) for c in NUM_SENSOR + ["ph"]})
    junk["soil_type"] = rng.choice(["Alluvial", "Black", "Red", "Laterite", "Sandy"], n)
    # physically inconsistent but in-range readings: e.g. sandy soil at 45 % moisture
    flag_test = (iso.predict(te[FEATURES_FULL]) == -1).mean()
    flag_junk = (iso.predict(junk[FEATURES_FULL]) == -1).mean()
    ood = pd.DataFrame([{"input": "real test readings", "n": len(te), "flagged_OOD": flag_test},
                        {"input": "random in-range readings (no real crop condition)", "n": n,
                         "flagged_OOD": flag_junk}])
    ood.round(4).to_csv(out / "ood_detection.csv", index=False)
    print(rc.round(3).to_string(index=False))
    t30 = rc[rc.threshold == 0.3].iloc[0]
    print(f"  conf>=0.30: coverage={t30.coverage:.3f} acc_accepted={t30.accuracy_on_accepted:.3f} "
          f"acc_flagged={t30.accuracy_on_flagged:.3f}")
    print(ood.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc.coverage, rc.accuracy_on_accepted, marker="o", color="#2e7d32")
    for _, r in rc.iloc[::3].iterrows():
        ax.annotate(f"t={r.threshold:.2f}", (r.coverage, r.accuracy_on_accepted), fontsize=7)
    ax.set_xlabel("coverage (share of recommendations shown as confident)")
    ax.set_ylabel("accuracy on confident recommendations"); ax.grid(alpha=.3)
    ax.set_title("Exp 5 - risk-coverage curve"); fig.tight_layout(); fig.savefig(out / "risk_coverage.png", dpi=150)
    plt.close(fig)
    return rc, ood


# ----------------------------------------------------------------------------- exp6
def exp6():
    out = EXP / "exp6_context_reranking"; out.mkdir(parents=True, exist_ok=True)
    rec = AgriRecommender()
    df = pd.read_csv(DATA)
    # pick the real cereal/fibre row on which the calibrated model is most torn
    # between its #1 crop and an N-fixing legume (context matters most there)
    best_i, best_margin = None, 9
    for i, r in df[df.label.isin(["rice", "maize", "cotton", "jute"])].iterrows():
        p = rec.predict_proba(pd.DataFrame([r]), "with_pH")[0]
        order = np.argsort(p)[::-1]; top = [rec.classes[j] for j in order[:3]]
        if ck.info(top[0])["family"] in ("Cereal", "Fibre") and any(ck.info(c)["family"] == "Legume" for c in top[1:]):
            leg = next(j for j in order[1:3] if ck.info(rec.classes[j])["family"] == "Legume")
            if p[order[0]] - p[leg] < best_margin:
                best_i, best_margin = i, p[order[0]] - p[leg]
    base = df.loc[best_i]
    print(f"  case-study reading: dataset row {best_i} (true label {base.label}), margin {best_margin:.3f}")
    readings = pd.DataFrame([{"log_ts": pd.Timestamp("2026-07-01") + pd.Timedelta(hours=4 * i),
                              **{c: base[c] for c in NUM_SENSOR}} for i in range(12)])
    scenarios = [
        ("S1 no history, July (Kharif)", [], 7),
        ("S2 last crop = model's #1 crop, July", "SAME", 7),
        ("S3 rice->maize->cotton, no fertiliser (N depleted), July",
         [{"crop": "cotton", "n_applied_kg_acre": 0}, {"crop": "maize", "n_applied_kg_acre": 0},
          {"crop": "rice", "n_applied_kg_acre": 0}], 7),
        ("S4 same history but 50 kg N/acre urea each cycle, July",
         [{"crop": "cotton", "n_applied_kg_acre": 50}, {"crop": "maize", "n_applied_kg_acre": 50},
          {"crop": "rice", "n_applied_kg_acre": 50}], 7),
        ("S5 no history, November (Rabi)", [], 11),
    ]
    first = rec.recommend(readings, base.soil_type, base.ph, [], 7, use_context=False).ranking[0]["crop"]
    rows = []
    for name, hist, month in scenarios:
        if hist == "SAME":
            hist = [{"crop": first, "n_applied_kg_acre": 0}]
        r_ml = rec.recommend(readings, base.soil_type, base.ph, hist, month, use_context=False)
        r_cx = rec.recommend(readings, base.soil_type, base.ph, hist, month, use_context=True)
        rows.append({"scenario": name, "VNBI": virtual_nitrogen_index(hist) if hist else None,
                     "ML-only top-3": ", ".join(f"{x['crop']} ({x['score']:.2f})" for x in r_ml.ranking),
                     "Context-aware top-3": ", ".join(f"{x['crop']} ({x['score']:.2f})" for x in r_cx.ranking),
                     "notes on #1": "; ".join(r_cx.ranking[0]["context_notes"]) or "-"})
    res = pd.DataFrame(rows)
    res.insert(0, "base_reading", f"row {best_i}: " + ", ".join(f"{c}={base[c]}" for c in NUM_SENSOR + ["ph", "soil_type"]))
    res.to_csv(out / "scenarios.csv", index=False)
    print(res.to_string(index=False))
    return res


# ----------------------------------------------------------------------------- summary
def md(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def write_summary(e3, e4, e5rc, e5ood, e6):
    e1 = pd.read_csv(EXP / "exp1_npk_ablation" / "results.csv").drop(columns=["fold_accuracies"])
    e2 = pd.read_csv(EXP / "exp2_model_comparison" / "summary_all_models.csv")
    e2s = e2[["feature_set", "model", "cv_accuracy", "cv_accuracy_std", "cv_f1_macro", "cv_top3_accuracy",
              "holdout_accuracy", "holdout_f1_macro", "holdout_top3_accuracy", "fit_seconds"]] \
        .sort_values(["feature_set", "cv_f1_macro"], ascending=[True, False])
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    txt = f"""# AgriSmart - Experimentation results (auto-generated)

Generated by `ml/train_models.py` and `ml/experiments.py` (seed {SEED}).
Every number below is produced by the code in this repository; re-running the
scripts regenerates this file. Dataset: `{meta['trained_on']}` ({meta['n_rows']} rows,
{len(meta['classes'])} crops), 80/20 stratified split, 5-fold stratified CV.

## Exp 0 - Dataset
See `exp0_dataset/` (summary statistics, per-crop statistics, crop x soil-type table,
correlation matrix, box plots).

## Exp 1 - Effect of removing the N-P-K sensor (original public dataset, Random Forest)
{md(e1)}

## Exp 2 - Model comparison on the AgriSmart sensor dataset
Per-model folders in `exp2_model_comparison/per_model/<feature_set>/<model>/` contain
`metrics.json`, `classification_report.csv`, `confusion_matrix.csv/.png`.
Per-fold scores: `exp2_model_comparison/cv_scores_per_fold.csv`.

{md(e2s)}

Selected (highest mean CV macro-F1):
* with pH    -> **{meta['models']['with_pH']['algorithm']}**
* without pH -> **{meta['models']['without_pH']['algorithm']}**

## Exp 3 - Novelty N1: Quality-Weighted Temporal Consensus under sensor faults
Top-1 accuracy on the test split (with-pH model, 42 readings / 7 days per test row).
Methods: A = classify the latest reading (conventional), B = latest range-valid reading,
C = plain mean of all readings, D = mean of per-reading probabilities (range-valid),
E = mean of range-valid readings, F = quality-weighted mean without recency,
G = quality x hourly recency (tau 72 h), H = proposed QW-TC (quality x day-blocked recency, tau 7 d).
Note: with n = 440 test rows the standard error of an accuracy near 0.85 is about
+/-0.017, so differences below ~3 points between E/F/H are within noise; the large
gain is snapshot (A/B) -> consensus, and quality weighting matters most at high fault rates.

{md(e3.pivot(index='fault_rate', columns='method', values='top1_accuracy').reset_index())}

Top-3 accuracy:

{md(e3.pivot(index='fault_rate', columns='method', values='top3_accuracy').reset_index())}

## Exp 4 - Novelty N2: adaptive model selection when pH is missing
{md(e4)}

## Exp 5 - Novelty N3: confidence-aware abstention
Risk-coverage (with-pH model, test split):

{md(e5rc)}

Out-of-distribution detector (Isolation Forest, contamination 1 %):

{md(e5ood)}

## Exp 6 - Novelty N4: context-aware re-ranking (case study, same soil readings)
{md(e6)}
"""
    (EXP / "EXPERIMENTS_SUMMARY.md").write_text(txt)
    print("wrote", (EXP / "EXPERIMENTS_SUMMARY.md").relative_to(ROOT))


def main():
    tr, te = split()
    models = fit_split_models(tr)
    print("== Exp 3: temporal consensus vs snapshot"); e3 = exp3(models, te)
    print("== Exp 4: adaptive model selection"); e4 = exp4(models, tr, te)
    print("== Exp 5: abstention"); e5rc, e5ood = exp5(models, tr, te)
    print("== Exp 6: context re-ranking"); e6 = exp6()
    write_summary(e3, e4, e5rc, e5ood, e6)


if __name__ == "__main__":
    main()
