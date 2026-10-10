"""
AgriSmart - Model comparison & training
=======================================
1. Compares 7 classifiers with stratified 5-fold cross-validation on the
   sensor dataset, for two feature sets (with pH / without pH).
2. Baseline experiment on the ORIGINAL public data: how much accuracy is lost
   when the N-P-K columns are removed (justifies the design choice).
3. Retrains the best model on an 80 % split, reports hold-out metrics,
   confusion matrix and permutation feature importance.
4. Saves the final pipelines + an out-of-distribution detector + learned
   per-crop "envelopes" (P5-P95 of each feature) used by the explainable
   recommender and by the MySQL crop catalogue.

Run:  python ml/train_models.py
"""
import json
import re
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              IsolationForest, RandomForestClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (DATA, FEATURES_FULL, FEATURES_NO_PH, MODEL_DIR, ROOT, SOIL_TYPES)

EXP_DIR = ROOT / "experiments"
REPORT_DIR = EXP_DIR

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "top3": "top_k_accuracy"}


def make_pre(features):
    num = [f for f in features if f != "soil_type"]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(categories=[SOIL_TYPES], handle_unknown="ignore",
                              sparse_output=False), ["soil_type"]),
    ])


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Gaussian Naive Bayes": GaussianNB(),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "SVM (RBF)": SVC(C=10, gamma="scale", probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
        # best model of Exp 7 (with pH: CV acc 0.895 vs HGB 0.888, better in 4/5 folds)
        "Soft Voting (RF+HGB+SVM)": VotingClassifier([
            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
            ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
            ("svm", SVC(C=10, probability=True, random_state=SEED))], voting="soft"),
    }


# Deployed model per feature set. None = pick the best by CV macro-F1.
# with pH: Soft Voting, chosen after Exp 7 (pH is entered manually from the Soil Health Card).
# without pH: Random Forest - in Exp 7 no ensemble beat it consistently (Soft Voting won 2/5 folds),
# so the simpler and faster model is kept as the fallback.
DEPLOY = {"with_pH": "Soft Voting (RF+HGB+SVM)", "without_pH": "Random Forest"}


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def compare(df, features, tag, models=None, out_root=None):
    """Evaluate EVERY candidate model and save its full evaluation:
    per-fold CV scores, hold-out metrics, per-crop classification report,
    confusion matrix (CSV + PNG) and train / inference time."""
    X, y = df[features], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    rows, folds = [], []
    models = models if models is not None else candidates()
    out_root = out_root or (EXP_DIR / "exp2_model_comparison" / "per_model")
    for name, clf in models.items():
        pipe = Pipeline([("pre", make_pre(features)), ("clf", clf)])
        t = time.time()
        s = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_sec = time.time() - t
        for k in range(len(s["test_accuracy"])):
            folds.append({"model": name, "feature_set": tag, "fold": k + 1,
                          "accuracy": s["test_accuracy"][k], "f1_macro": s["test_f1_macro"][k],
                          "top3_accuracy": s["test_top3"][k]})

        # hold-out evaluation of this model
        t = time.time(); pipe.fit(Xtr, ytr); fit_sec = time.time() - t
        t = time.time(); proba = pipe.predict_proba(Xte); pred_ms = (time.time() - t) / len(Xte) * 1000
        pred = pipe.classes_[proba.argmax(1)]
        out = out_root / tag / safe(name); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(classification_report(yte, pred, output_dict=True, zero_division=0)).T.round(4) \
            .to_csv(out / "classification_report.csv")
        cm = confusion_matrix(yte, pred, labels=pipe.classes_)
        pd.DataFrame(cm, index=pipe.classes_, columns=pipe.classes_).to_csv(out / "confusion_matrix.csv")
        plot_confusion(yte, pred, list(pipe.classes_), out / "confusion_matrix.png",
                       f"{name} ({tag.replace('_', ' ')}) - hold-out")
        row = {
            "model": name, "feature_set": tag,
            "cv_accuracy": s["test_accuracy"].mean(), "cv_accuracy_std": s["test_accuracy"].std(),
            "cv_f1_macro": s["test_f1_macro"].mean(), "cv_f1_macro_std": s["test_f1_macro"].std(),
            "cv_top3_accuracy": s["test_top3"].mean(),
            "holdout_accuracy": accuracy_score(yte, pred),
            "holdout_precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
            "holdout_recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
            "holdout_f1_macro": f1_score(yte, pred, average="macro"),
            "holdout_top3_accuracy": top_k_accuracy_score(yte, proba, k=3, labels=pipe.classes_),
            "holdout_log_loss": log_loss(yte, proba, labels=pipe.classes_),
            "fit_seconds": round(fit_sec, 3), "cv_seconds": round(cv_sec, 2),
            "predict_ms_per_sample": round(pred_ms, 4),
        }
        (out / "metrics.json").write_text(json.dumps(row, indent=2, default=float))
        rows.append(row)
        print(f"  [{tag}] {name:24s} cv_acc={row['cv_accuracy']:.4f} cv_f1={row['cv_f1_macro']:.4f} "
              f"hold_acc={row['holdout_accuracy']:.4f} top3={row['holdout_top3_accuracy']:.4f}")
    return pd.DataFrame(rows), pd.DataFrame(folds)


def npk_ablation():
    """Experiment on the ORIGINAL public dataset: effect of dropping N-P-K."""
    raw = pd.read_csv(ROOT / "data" / "raw" / "crop_recommendation.csv")
    sets = {
        "N,P,K,temp,humidity,pH,rainfall (original)": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        "temp,humidity,pH,rainfall (no NPK)": ["temperature", "humidity", "ph", "rainfall"],
        "temp,humidity,rainfall (no NPK, no pH)": ["temperature", "humidity", "rainfall"],
        "temp,humidity,pH (no NPK, no rainfall)": ["temperature", "humidity", "ph"],
    }
    out = []
    for name, cols in sets.items():
        rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1)
        s = cross_validate(rf, raw[cols], raw["label"], cv=CV, scoring=SCORING)
        out.append({"features": name, "model": "Random Forest (400 trees)",
                    "accuracy": s["test_accuracy"].mean(), "accuracy_std": s["test_accuracy"].std(),
                    "f1_macro": s["test_f1_macro"].mean(), "top3_accuracy": s["test_top3"].mean(),
                    "fold_accuracies": ";".join(f"{v:.4f}" for v in s["test_accuracy"])})
    return pd.DataFrame(out)


def ece(proba, y, classes, bins=15):
    """Expected calibration error of the top-1 prediction."""
    conf = proba.max(1); pred = np.asarray(classes)[proba.argmax(1)]; acc = (pred == y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def envelopes(df):
    """Learned P5-P95 range of every numeric feature for each crop."""
    num = [f for f in FEATURES_FULL if f != "soil_type"]
    env = {}
    for crop, g in df.groupby("label"):
        env[crop] = {f: [round(float(g[f].quantile(.05)), 2), round(float(g[f].quantile(.95)), 2)]
                     for f in num}
        env[crop]["soil_types"] = g["soil_type"].value_counts(normalize=True).round(3).to_dict()
    return env


def plot_confusion(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def eda(df):
    """Dataset description figures/tables for the report."""
    out = EXP_DIR / "exp0_dataset"; out.mkdir(parents=True, exist_ok=True)
    df.describe().round(3).to_csv(out / "summary_statistics.csv")
    df.groupby("label")[[f for f in FEATURES_FULL if f != "soil_type"]].agg(["mean", "std", "min", "max"]) \
        .round(2).to_csv(out / "per_crop_statistics.csv")
    pd.crosstab(df.label, df.soil_type).to_csv(out / "crop_vs_soil_type.csv")
    df[[f for f in FEATURES_FULL if f != "soil_type"]].corr().round(3).to_csv(out / "correlation_matrix.csv")
    num = [f for f in FEATURES_FULL if f != "soil_type"]
    fig, axes = plt.subplots(len(num), 1, figsize=(12, 2.4 * len(num)), sharex=True)
    order = sorted(df.label.unique())
    for ax, f in zip(axes, num):
        ax.boxplot([df.loc[df.label == c, f] for c in order], showfliers=False)
        ax.set_ylabel(f, fontsize=8)
    axes[-1].set_xticks(range(1, len(order) + 1), order, rotation=90, fontsize=8)
    fig.suptitle("Feature distribution per crop (AgriSmart sensor dataset)")
    fig.tight_layout(); fig.savefig(out / "feature_boxplots_per_crop.png", dpi=130); plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True); EXP_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    eda(df)

    print("== Experiment 1: N-P-K ablation on the original public dataset")
    abl = npk_ablation(); print(abl.round(4).to_string(index=False))
    out1 = EXP_DIR / "exp1_npk_ablation"; out1.mkdir(exist_ok=True)
    abl.to_csv(out1 / "results.csv", index=False)

    print("== Experiment 2: model comparison on the AgriSmart sensor dataset")
    out2 = EXP_DIR / "exp2_model_comparison"; out2.mkdir(exist_ok=True)
    r1, f1 = compare(df, FEATURES_FULL, "with_pH")
    r2, f2 = compare(df, FEATURES_NO_PH, "without_pH")
    cmp_df = pd.concat([r1, r2]); cmp_df.round(4).to_csv(out2 / "summary_all_models.csv", index=False)
    pd.concat([f1, f2]).round(4).to_csv(out2 / "cv_scores_per_fold.csv", index=False)

    meta = {"trained_on": str(DATA.relative_to(ROOT)), "n_rows": len(df),
            "classes": sorted(df.label.unique()), "seed": SEED, "models": {}}

    for tag, feats in [("with_pH", FEATURES_FULL), ("without_pH", FEATURES_NO_PH)]:
        sub = cmp_df[cmp_df.feature_set == tag].sort_values(["cv_f1_macro", "cv_accuracy"], ascending=False)
        if DEPLOY.get(tag):
            sub = pd.concat([sub[sub.model == DEPLOY[tag]], sub[sub.model != DEPLOY[tag]]])
            best = DEPLOY[tag]
            print(f"== Deployed for {tag}: {best}  (fixed in DEPLOY; best by CV macro-F1 was "
                  f"{cmp_df[cmp_df.feature_set == tag].sort_values('cv_f1_macro').iloc[-1].model})")
        else:
            best = sub.iloc[0]["model"]
            print(f"== Best for {tag}: {best}  (selected by mean 5-fold CV macro-F1)")

        X, y = df[feats], df["label"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        pipe = Pipeline([("pre", make_pre(feats)), ("clf", candidates()[best])]).fit(Xtr, ytr)
        pi = permutation_importance(pipe, Xte, yte, n_repeats=15, random_state=SEED, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=feats).sort_values()
        imp.to_csv(out2 / f"feature_importance_{tag}.csv", header=["importance"])
        fig, ax = plt.subplots(figsize=(6, 3.5)); imp.plot.barh(ax=ax, color="#2e7d32")
        ax.set_title(f"Permutation importance - {best} ({tag.replace('_', ' ')})", fontsize=9)
        ax.set_xlabel("mean drop in accuracy")
        fig.tight_layout(); fig.savefig(out2 / f"feature_importance_{tag}.png", dpi=150); plt.close(fig)

        # ---- probability calibration (needed for abstention + re-ranking) ----
        cal = CalibratedClassifierCV(Pipeline([("pre", make_pre(feats)), ("clf", candidates()[best])]),
                                     method="isotonic", cv=5).fit(Xtr, ytr)
        crow = []
        for nm, mdl in [("uncalibrated", pipe), ("isotonic-calibrated", cal)]:
            pr = mdl.predict_proba(Xte); pd_ = mdl.classes_[pr.argmax(1)]
            crow.append({"model": f"{best} ({nm})", "holdout_accuracy": accuracy_score(yte, pd_),
                         "holdout_top3_accuracy": top_k_accuracy_score(yte, pr, k=3, labels=mdl.classes_),
                         "log_loss": log_loss(yte, pr, labels=mdl.classes_),
                         "ECE_15bins": ece(pr, yte.to_numpy(), mdl.classes_),
                         "mean_top1_confidence": pr.max(1).mean()})
        cdf = pd.DataFrame(crow); cdf.round(4).to_csv(out2 / f"calibration_{tag}.csv", index=False)
        print(cdf.round(4).to_string(index=False))

        # final (calibrated) model on ALL data for deployment
        final = CalibratedClassifierCV(Pipeline([("pre", make_pre(feats)), ("clf", candidates()[best])]),
                                       method="isotonic", cv=5).fit(X, y)
        joblib.dump(final, MODEL_DIR / f"crop_model_{tag}.joblib", compress=3)
        # Out-of-distribution detector (novelty N3) - has its own preprocessor
        ood = Pipeline([("pre", make_pre(feats)),
                        ("iso", IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED))]).fit(X)
        joblib.dump(ood, MODEL_DIR / f"ood_detector_{tag}.joblib", compress=3)

        b = sub.iloc[0]
        meta["models"][tag] = {"algorithm": best, "features": feats,
                               "cv_accuracy": b.cv_accuracy, "cv_f1_macro": b.cv_f1_macro,
                               "cv_top3_accuracy": b.cv_top3_accuracy,
                               "holdout_accuracy": b.holdout_accuracy, "holdout_f1_macro": b.holdout_f1_macro,
                               "holdout_top3_accuracy": b.holdout_top3_accuracy}

    for metric, label in [("cv_accuracy", "5-fold CV accuracy"), ("cv_f1_macro", "5-fold CV macro-F1"),
                          ("holdout_top3_accuracy", "hold-out top-3 accuracy")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        piv = cmp_df.pivot(index="model", columns="feature_set", values=metric).sort_values("with_pH")
        piv[["without_pH", "with_pH"]].plot.barh(ax=ax, color=["#a5d6a7", "#2e7d32"]); ax.set_xlim(0, 1)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xlabel(label); ax.set_title(f"Model comparison - {label}")
        fig.tight_layout(); fig.savefig(out2 / f"chart_{metric}.png", dpi=150); plt.close(fig)

    meta["envelopes"] = envelopes(df)
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print("saved models + meta to", MODEL_DIR.relative_to(ROOT), "and evaluations to", EXP_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
