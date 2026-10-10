"""
Exp 7 - Extended model comparison (every model in ml/model_zoo.py)
=================================================================
    python ml/compare_models.py              # baseline + extra models, both feature sets
    python ml/compare_models.py --only-extra # just the EXTRA_MODELS
Output: experiments/exp7_extended_model_comparison/
    summary_all_models.csv, cv_scores_per_fold.csv, per_model/<feature_set>/<model>/...,
    chart_*.png (accuracy, macro-F1, top-3, accuracy-vs-training-time)
Does NOT change the deployed models (that is train_models.py's job).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES_FULL, FEATURES_NO_PH, ROOT  # noqa: E402
from model_zoo import BASELINE_MODELS, EXTRA_MODELS, all_models  # noqa: E402
from train_models import compare  # noqa: E402

OUT = ROOT / "experiments" / "exp7_extended_model_comparison"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-extra", action="store_true")
    a = ap.parse_args()
    models = EXTRA_MODELS if a.only_extra else all_models()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"Comparing {len(models)} models: {', '.join(models)}")
    rs, fs = [], []
    for tag, feats in (("with_pH", FEATURES_FULL), ("without_pH", FEATURES_NO_PH)):
        r, f = compare(df, feats, tag, models=models, out_root=OUT / "per_model")
        rs.append(r); fs.append(f)
    res = pd.concat(rs); res["group"] = res.model.map(lambda m: "baseline" if m in BASELINE_MODELS else "extra")
    res.round(4).to_csv(OUT / "summary_all_models.csv", index=False)
    pd.concat(fs).round(4).to_csv(OUT / "cv_scores_per_fold.csv", index=False)

    for metric, label in [("cv_accuracy", "5-fold CV accuracy"), ("cv_f1_macro", "5-fold CV macro-F1"),
                          ("holdout_top3_accuracy", "hold-out top-3 accuracy")]:
        piv = res.pivot(index="model", columns="feature_set", values=metric).sort_values("with_pH")
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(piv) + 1.5))
        piv[["without_pH", "with_pH"]].plot.barh(ax=ax, color=["#a5d6a7", "#2e7d32"])
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xlim(0, 1.08); ax.set_xlabel(label); ax.set_title(f"Extended model comparison - {label}")
        fig.tight_layout(); fig.savefig(OUT / f"chart_{metric}.png", dpi=150); plt.close(fig)

    sub = res[res.feature_set == "with_pH"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, col in (("baseline", "#2e7d32"), ("extra", "#b5652b")):
        s = sub[sub.group == g]
        ax.scatter(s.fit_seconds, s.cv_accuracy, s=60, color=col, label=g)
        for _, r in s.iterrows():
            ax.annotate(r.model, (r.fit_seconds, r.cv_accuracy), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("training time on 80 % split (s, log scale)"); ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Accuracy vs training cost (with pH)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "chart_accuracy_vs_training_time.png", dpi=150); plt.close(fig)
    print(res.sort_values(["feature_set", "cv_f1_macro"], ascending=[True, False])
          [["feature_set", "model", "cv_accuracy", "cv_f1_macro", "holdout_top3_accuracy", "fit_seconds"]]
          .round(4).to_string(index=False))
    print("saved to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
