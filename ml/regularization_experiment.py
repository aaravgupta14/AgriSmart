"""
Exp 8 - Does regularization improve accuracy?
=============================================
    python ml/regularization_experiment.py

For each model, the regularization strength is chosen by an INNER 3-fold grid
search and scored on the SAME OUTER 5-fold split as Exp 2 / Exp 7 (nested CV),
so the tuned score is not optimistically biased. The per-fold difference to the
untuned (default) model is reported.

A second, descriptive sweep (plain 5-fold CV, one parameter at a time) shows how
accuracy changes with regularization strength - for figures only, not for
picking the model.

Output: experiments/exp8_regularization/
    nested_cv_summary.csv, nested_cv_per_fold.csv, chosen_params.csv,
    sweep.csv, sweep_<model>.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES_FULL, FEATURES_NO_PH, ROOT  # noqa: E402
from train_models import CV, SEED, make_pre  # noqa: E402

OUT = ROOT / "experiments" / "exp8_regularization"
INNER = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

# model -> (default classifier, regularization grid). Defaults match Exp 2 / model_zoo.
MODELS = {
    "Hist Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"l2_regularization": [0.0, 0.1, 1.0, 10.0],
         "min_samples_leaf": [20, 50],
         "max_leaf_nodes": [15, 31]}),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
        {"min_samples_leaf": [1, 3, 5],
         "max_features": ["sqrt", 0.5],
         "max_depth": [None, 15]}),
    "SVM (RBF)": (
        SVC(C=10, gamma="scale", probability=True, random_state=SEED),
        {"C": [0.3, 1, 3, 10, 30, 100]}),
    "Logistic Regression": (
        LogisticRegression(max_iter=3000),
        {"C": [0.01, 0.1, 1, 10, 100]}),
    "MLP Neural Network": (
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=SEED),
        {"alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}),
}

# one parameter per model for the descriptive sweep (larger value = stronger regularization,
# except C where smaller = stronger)
SWEEP = {
    "Hist Gradient Boosting": ("l2_regularization", [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]),
    "Random Forest": ("min_samples_leaf", [1, 2, 3, 5, 8, 12]),
    "SVM (RBF)": ("C", [0.1, 0.3, 1, 3, 10, 30, 100]),
    "Logistic Regression": ("C", [0.001, 0.01, 0.1, 1, 10, 100]),
    "MLP Neural Network": ("alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
}


def pipe(clf, features):
    # clone: set_params on the pipeline must never modify the shared default in MODELS
    return Pipeline([("pre", make_pre(features)), ("clf", clone(clf))])


def nested(df, features, tag):
    X, y = df[features], df["label"]
    summary, folds, chosen = [], [], []
    for name, (clf, grid) in MODELS.items():
        t = time.time()
        base = cross_val_score(pipe(clf, features), X, y, cv=CV, scoring="accuracy", n_jobs=-1)
        gs = GridSearchCV(pipe(clf, features), {f"clf__{k}": v for k, v in grid.items()},
                          cv=INNER, scoring="accuracy", n_jobs=-1)
        tuned = []
        for k, (tr, te) in enumerate(CV.split(X, y), 1):
            gs.fit(X.iloc[tr], y.iloc[tr])
            acc = (gs.predict(X.iloc[te]) == y.iloc[te]).mean()
            tuned.append(acc)
            chosen.append({"model": name, "feature_set": tag, "fold": k,
                           **{p.replace("clf__", ""): v for p, v in gs.best_params_.items()}})
            folds.append({"model": name, "feature_set": tag, "fold": k,
                          "default_accuracy": base[k - 1], "regularized_accuracy": acc,
                          "diff": acc - base[k - 1]})
        tuned = np.array(tuned); d = tuned - base
        summary.append({"model": name, "feature_set": tag,
                        "default_cv_accuracy": base.mean(), "default_std": base.std(),
                        "regularized_cv_accuracy": tuned.mean(), "regularized_std": tuned.std(),
                        "mean_gain": d.mean(), "folds_improved": int((d > 0).sum()),
                        "folds_worse": int((d < 0).sum()), "seconds": round(time.time() - t, 1)})
        print(f"  [{tag}] {name:24s} default={base.mean():.4f}  regularized={tuned.mean():.4f}  "
              f"gain={d.mean():+.4f}  improved {int((d > 0).sum())}/5 folds")
    return summary, folds, chosen


def sweep(df, features, tag):
    X, y = df[features], df["label"]
    rows = []
    for name, (param, values) in SWEEP.items():
        clf = MODELS[name][0]
        for v in values:
            s = cross_val_score(pipe(clf, features).set_params(**{f"clf__{param}": v}), X, y,
                                cv=CV, scoring="accuracy", n_jobs=-1)
            rows.append({"model": name, "feature_set": tag, "param": param, "value": v,
                         "cv_accuracy": s.mean(), "cv_std": s.std()})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    S, F, C, W = [], [], [], []
    for tag, feats in (("with_pH", FEATURES_FULL), ("without_pH", FEATURES_NO_PH)):
        print(f"== nested CV, {tag}")
        s, f, c = nested(df, feats, tag); S += s; F += f; C += c
        print(f"== sweep, {tag}")
        W += sweep(df, feats, tag)
    pd.DataFrame(S).round(4).to_csv(OUT / "nested_cv_summary.csv", index=False)
    pd.DataFrame(F).round(4).to_csv(OUT / "nested_cv_per_fold.csv", index=False)
    pd.DataFrame(C).to_csv(OUT / "chosen_params.csv", index=False)
    w = pd.DataFrame(W); w.round(4).to_csv(OUT / "sweep.csv", index=False)

    for name, (param, _) in SWEEP.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for tag, color in (("with_pH", "#2e7d32"), ("without_pH", "#a5d6a7")):
            g = w[(w.model == name) & (w.feature_set == tag)]
            ax.errorbar(g.value, g.cv_accuracy, yerr=g.cv_std, marker="o", capsize=3, color=color, label=tag)
        ax.set_xscale("symlog" if param in ("l2_regularization",) else ("linear" if param == "min_samples_leaf" else "log"),
                      **({"linthresh": 0.01} if param == "l2_regularization" else {}))
        ax.set_xlabel(param); ax.set_ylabel("5-fold CV accuracy")
        ax.set_title(f"{name}: accuracy vs {param}"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / f"sweep_{name.split(' (')[0].lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
    print(f"saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
