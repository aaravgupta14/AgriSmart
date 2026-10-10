"""
Collects / generates every report figure into experiments/figures/
==================================================================
Run AFTER train_models.py and experiments.py (and optionally compare_models.py):
    python ml/make_figures.py
Creates numbered PNGs (150 dpi) + experiments/figures/FIGURES_INDEX.md with a
caption for each figure, and copies all 16+ per-model confusion matrices into
experiments/figures/confusion_matrices/.
"""
import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES_FULL, ROOT  # noqa: E402

EXP = ROOT / "experiments"
FIG = EXP / "figures"
GREEN, LIGHT, CLAY, INK = "#2e7d32", "#a5d6a7", "#b5652b", "#2a2420"
index = []


def save(fig, name, caption):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150); plt.close(fig)
    index.append((name, caption)); print("  ", name)


def copy(src, name, caption):
    src = EXP / src
    if src.exists():
        shutil.copy(src, FIG / name); index.append((name, caption)); print("  ", name)
    else:
        print("   (missing)", src.relative_to(ROOT))


def heat(ax, M, rows, cols, fmt="{:.2f}", cmap="Greens", vmin=None, vmax=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8); ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (vmax or M.max()) and abs(v) > 0.6 * (vmax or M.max()) else "black")
    return im


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    num = [f for f in FEATURES_FULL if f != "soil_type"]

    # ---------------------------------------------------------------- dataset
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    df.label.value_counts().sort_index().plot.bar(ax=axes[0], color=GREEN)
    axes[0].set_title("Samples per crop (balanced)"); axes[0].set_ylabel("rows")
    df.soil_type.value_counts().plot.bar(ax=axes[1], color=CLAY)
    axes[1].set_title("Samples per soil type"); axes[1].set_ylabel("rows")
    save(fig, "fig01_class_and_soil_distribution.png", "Class balance (100 rows per crop) and soil-type distribution of the AgriSmart sensor dataset.")
    copy("exp0_dataset/feature_boxplots_per_crop.png", "fig02_feature_boxplots_per_crop.png",
         "Distribution of each sensor feature per crop (box plots, outliers hidden).")
    fig, ax = plt.subplots(figsize=(6, 5))
    C = df[num].corr().to_numpy()
    im = heat(ax, C, num, num, cmap="RdBu_r", vmin=-1, vmax=1); fig.colorbar(im, fraction=0.046)
    ax.set_title("Pearson correlation between features")
    save(fig, "fig03_feature_correlation_heatmap.png", "Correlation matrix of the five numeric features.")
    ct = pd.crosstab(df.label, df.soil_type, normalize="index")
    fig, ax = plt.subplots(figsize=(6, 8))
    heat(ax, ct.to_numpy(), list(ct.index), list(ct.columns))
    ax.set_title("Share of each soil type per crop")
    save(fig, "fig04_crop_vs_soil_type_heatmap.png", "Crop-soil association used to assign soil types (row-normalised).")

    # ------------------------------------------------------------- exp 1
    e1 = pd.read_csv(EXP / "exp1_npk_ablation" / "results.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(e1.features, e1.accuracy, color=[GREEN] + [LIGHT] * (len(e1) - 1),
                   xerr=e1.accuracy_std, capsize=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.08)
    ax.set_xlabel("5-fold CV accuracy (Random Forest)"); ax.set_title("Exp 1 - accuracy lost without the NPK sensor")
    save(fig, "fig05_exp1_npk_ablation.png", "Exp 1: removing N, P, K (and rainfall/pH) from the original public dataset.")

    # ------------------------------------------------------------- exp 2
    copy("exp2_model_comparison/chart_cv_accuracy.png", "fig06_exp2_model_comparison_accuracy.png", "Exp 2: 5-fold CV accuracy of 8 classifiers, with vs without pH.")
    copy("exp2_model_comparison/chart_cv_f1_macro.png", "fig07_exp2_model_comparison_f1.png", "Exp 2: 5-fold CV macro-F1.")
    copy("exp2_model_comparison/chart_holdout_top3_accuracy.png", "fig08_exp2_model_comparison_top3.png", "Exp 2: hold-out top-3 accuracy (the website shows a top-3).")
    folds = pd.read_csv(EXP / "exp2_model_comparison" / "cv_scores_per_fold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, tag in zip(axes, ["with_pH", "without_pH"]):
        f = folds[folds.feature_set == tag]
        order = f.groupby("model").accuracy.mean().sort_values().index
        ax.boxplot([f[f.model == m].accuracy for m in order], vert=False, patch_artist=True,
                   boxprops=dict(facecolor=LIGHT))
        ax.set_yticks(range(1, len(order) + 1), order, fontsize=8); ax.set_title(tag.replace("_", " "))
        ax.set_xlabel("fold accuracy"); ax.grid(alpha=.3)
    fig.suptitle("Exp 2 - spread of accuracy across the 5 CV folds")
    save(fig, "fig09_exp2_cv_fold_spread.png", "Exp 2: per-fold accuracy spread (stability of each model).")

    import json
    meta = json.loads((ROOT / "ml" / "models" / "model_meta.json").read_text())
    safe = lambda n: re.sub(r"[^A-Za-z0-9]+", "_", n).strip("_").lower()
    for k, tag in ((10, "with_pH"), (11, "without_pH")):
        best = meta["models"][tag]["algorithm"]
        copy(f"exp2_model_comparison/per_model/{tag}/{safe(best)}/confusion_matrix.png",
             f"fig{k}_confusion_matrix_best_{tag}.png", f"Hold-out confusion matrix of the selected model ({best}, {tag.replace('_', ' ')}).")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = .4
    for i, tag in enumerate(["with_pH", "without_pH"]):
        best = meta["models"][tag]["algorithm"]
        rep = pd.read_csv(EXP / "exp2_model_comparison" / "per_model" / tag / safe(best) / "classification_report.csv", index_col=0)
        rep = rep.drop(index=[r for r in rep.index if r in ("accuracy", "macro avg", "weighted avg")])
        x = np.arange(len(rep))
        ax.bar(x + (i - .5) * width, rep["f1-score"], width, color=[GREEN, LIGHT][i], label=f"{best} ({tag.replace('_', ' ')})")
    ax.set_xticks(x, rep.index, rotation=90); ax.set_ylabel("F1-score (hold-out)"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-crop F1-score of the selected models"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    save(fig, "fig12_per_crop_f1.png", "Per-crop F1 on the hold-out set - shows which crops are hard to separate without NPK.")
    copy("exp2_model_comparison/feature_importance_with_pH.png", "fig13_feature_importance_with_pH.png", "Permutation feature importance (with pH).")
    copy("exp2_model_comparison/feature_importance_without_pH.png", "fig14_feature_importance_without_pH.png", "Permutation feature importance (without pH).")

    # calibration + learning curve (computed here)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import learning_curve, train_test_split
    from sklearn.pipeline import Pipeline
    from train_models import CV, candidates, make_pre
    best = meta["models"]["with_pH"]["algorithm"]
    X, y = df[FEATURES_FULL], df.label
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    raw = Pipeline([("pre", make_pre(FEATURES_FULL)), ("clf", candidates()[best])]).fit(Xtr, ytr)
    cal = CalibratedClassifierCV(Pipeline([("pre", make_pre(FEATURES_FULL)), ("clf", candidates()[best])]),
                                 method="isotonic", cv=5).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    for mdl, nm, col in ((raw, "uncalibrated", CLAY), (cal, "isotonic-calibrated", GREEN)):
        P = mdl.predict_proba(Xte); conf = P.max(1); ok = mdl.classes_[P.argmax(1)] == yte.to_numpy()
        bins = np.linspace(0, 1, 11); idx = np.digitize(conf, bins[1:-1])
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [ok[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ax.plot(xs, ys, "o-", color=col, label=nm)
    ax.set_xlabel("predicted confidence (top-1)"); ax.set_ylabel("observed accuracy"); ax.legend(fontsize=8)
    ax.set_title(f"Reliability diagram - {best}"); ax.grid(alpha=.3)
    save(fig, "fig15_reliability_diagram.png", "Reliability diagram before/after isotonic calibration (novelty N3).")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nm, col in ((best, GREEN), ("Random Forest", CLAY), ("K-Nearest Neighbours", "#6f8fb3")):
        pipe = Pipeline([("pre", make_pre(FEATURES_FULL)), ("clf", candidates()[nm])])
        sizes, trs, tes = learning_curve(pipe, X, y, cv=CV, train_sizes=np.linspace(.1, 1, 6), n_jobs=-1)
        ax.plot(sizes, tes.mean(1), "o-", color=col, label=f"{nm} (validation)")
        ax.fill_between(sizes, tes.mean(1) - tes.std(1), tes.mean(1) + tes.std(1), color=col, alpha=.15)
        ax.plot(sizes, trs.mean(1), "--", color=col, alpha=.6, lw=1)
    ax.set_xlabel("training rows"); ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.02); ax.grid(alpha=.3)
    ax.set_title("Learning curves (dashed = training accuracy)"); ax.legend(fontsize=8)
    save(fig, "fig16_learning_curves.png", "Learning curves: validation accuracy still rising -> more (field) data would help.")

    # ------------------------------------------------------------- exp 3-6
    copy("exp3_temporal_consensus/chart.png", "fig17_exp3_temporal_consensus_top1.png", "Exp 3 (novelty N1): top-1 accuracy vs sensor fault rate for 8 aggregation methods.")
    e3 = pd.read_csv(EXP / "exp3_temporal_consensus" / "results.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sel = ["A_snapshot_raw", "D_prob_mean_validated", "E_feature_mean_validated", "H_QW-TC (proposed)"]
    for i, m in enumerate(sel):
        s = e3[e3.method == m]
        ax.bar(np.arange(len(s)) + (i - 1.5) * .2, s.top1_accuracy, .2, label=m,
               color=["#c9c0b3", LIGHT, "#6f8fb3", GREEN][i])
    ax.set_xticks(range(len(s)), [f"{int(f * 100)} %" for f in s.fault_rate]); ax.set_xlabel("fault rate per reading")
    ax.set_ylabel("top-1 accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
    ax.set_title("Exp 3 - snapshot vs temporal consensus")
    save(fig, "fig18_exp3_snapshot_vs_consensus_bars.png", "Exp 3: conventional snapshot vs proposed QW-TC (bar view).")
    e4 = pd.read_csv(EXP / "exp4_adaptive_model_selection" / "results.csv")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    cols = [GREEN if "proposed" in s else ("#c9c0b3" if "measured" in sc else CLAY) for s, sc in zip(e4.strategy, e4.scenario)]
    b = ax.barh(e4.scenario + " | " + e4.strategy, e4.accuracy, color=cols)
    ax.bar_label(b, fmt="%.3f", padding=3, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("hold-out accuracy"); ax.set_title("Exp 4 - what to do when pH is missing")
    save(fig, "fig19_exp4_adaptive_model_selection.png", "Exp 4 (novelty N2): adaptive switch to the sensor-only model vs imputing pH.")
    copy("exp5_abstention/risk_coverage.png", "fig20_exp5_risk_coverage.png", "Exp 5 (novelty N3): accuracy of recommendations shown as confident vs coverage.")
    ood = pd.read_csv(EXP / "exp5_abstention" / "ood_detection.csv")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    b = ax.barh(ood.input, ood.flagged_OOD * 100, color=[GREEN, CLAY]); ax.bar_label(b, fmt="%.1f %%", padding=3)
    ax.set_xlim(0, 100); ax.set_xlabel("% flagged as out-of-distribution"); ax.invert_yaxis()
    ax.set_title("Exp 5 - Isolation-Forest OOD detector")
    save(fig, "fig21_exp5_ood_detection.png", "Exp 5: OOD detector flags implausible inputs but almost no real ones.")
    e6 = pd.read_csv(EXP / "exp6_context_reranking" / "scenarios.csv")
    parse = lambda s: {m.group(1): float(m.group(2)) for m in re.finditer(r"([a-z]+) \(([\d.]+)\)", s)}
    crops = list(parse(e6["ML-only top-3"].iloc[0]))[:2]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(e6))
    for i, c in enumerate(crops):
        ax.bar(x + (i - .5) * .35, [parse(s).get(c, 0) for s in e6["Context-aware top-3"]], .35,
               color=[CLAY, GREEN][i], label=f"{c} (context-aware score)")
        ax.plot(x, [parse(e6["ML-only top-3"].iloc[0]).get(c, 0)] * len(x), ":", color=[CLAY, GREEN][i])
    ax.set_xticks(x, [s.split(" ", 1)[0] for s in e6.scenario]); ax.set_ylabel("final score")
    ax.set_title("Exp 6 - same soil readings, different plot histories (dotted = ML-only)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    save(fig, "fig22_exp6_context_reranking.png", "Exp 6 (novelty N4): rotation / season / VNBI re-ranking. S1-S5 are described in EXPERIMENTS_SUMMARY.md.")

    # ------------------------------------------------------------- exp 7 (optional)
    for k, (f, cap) in enumerate([("chart_cv_accuracy.png", "Exp 7: extended comparison, all models in model_zoo.py - CV accuracy."),
                                   ("chart_cv_f1_macro.png", "Exp 7: extended comparison - CV macro-F1."),
                                   ("chart_holdout_top3_accuracy.png", "Exp 7: extended comparison - hold-out top-3 accuracy."),
                                   ("chart_accuracy_vs_training_time.png", "Exp 7: accuracy vs training cost.")]):
        copy(f"exp7_extended_model_comparison/{f}", f"fig{23 + k}_exp7_{f.replace('chart_', '')}", cap)

    # ------------------------------------------------------------- all confusion matrices
    cm = FIG / "confusion_matrices"; cm.mkdir(exist_ok=True); n = 0
    for exp in ("exp2_model_comparison", "exp7_extended_model_comparison"):
        for p in (EXP / exp / "per_model").glob("*/*/confusion_matrix.png"):
            shutil.copy(p, cm / f"{exp.split('_')[0]}_{p.parent.parent.name}_{p.parent.name}.png"); n += 1
    print(f"   + {n} confusion matrices in figures/confusion_matrices/")

    md = ["# Figures for the report (auto-generated by ml/make_figures.py)", "",
          "| File | Caption |", "|---|---|"] + [f"| `{f}` | {c} |" for f, c in index]
    md += ["", f"Plus `confusion_matrices/` - hold-out confusion matrix of every model ({n} files)."]
    (FIG / "FIGURES_INDEX.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", (FIG / "FIGURES_INDEX.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
