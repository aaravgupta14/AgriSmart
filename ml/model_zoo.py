"""
AgriSmart MODEL ZOO - the one place to add new models for experimentation.
=========================================================================
HOW TO ADD A MODEL
  1. import it below
  2. add ONE line to EXTRA_MODELS:   "My Model Name": MyClassifier(params...),
  3. run:   python ml/compare_models.py
Results go to experiments/exp7_extended_model_comparison/ (same metrics,
classification report and confusion matrix per model as Exp 2, plus charts).

Every model must be a scikit-learn compatible classifier with predict_proba
(SVC needs probability=True). Preprocessing (scaling + one-hot soil type) is
added automatically, so put only the classifier here.

Optional libraries (XGBoost, LightGBM, CatBoost) are added automatically if
they are installed:   pip install xgboost lightgbm catboost
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# --- the 8 baseline models used in Exp 2 (kept identical to train_models.py) ---
BASELINE_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "SVM (RBF)": SVC(C=10, gamma="scale", probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
}

# --- additional models for experimentation: ADD NEW ONES HERE ---------------
EXTRA_MODELS = {
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                        early_stopping=True, random_state=SEED),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "AdaBoost (trees)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                                           learning_rate=0.5, random_state=SEED),
    "Bagging (trees)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200,
                                         random_state=SEED, n_jobs=-1),
    "Gradient Boosting (classic)": GradientBoostingClassifier(n_estimators=150, random_state=SEED),
    "Soft Voting (RF+HGB+SVM)": VotingClassifier([
        ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
        ("svm", SVC(C=10, probability=True, random_state=SEED))], voting="soft"),
    "Stacking (RF+HGB+KNN -> LR)": StackingClassifier([
        ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))],
        final_estimator=LogisticRegression(max_iter=3000), cv=5),
}

# --- optional third-party boosters (used only if installed) -----------------
try:
    from xgboost import XGBClassifier  # labels are strings -> wrapped below
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder

    class XGBStringLabels(BaseEstimator, ClassifierMixin):
        """XGBoost needs integer labels; this wrapper accepts crop names."""
        def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1):
            self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate

        def fit(self, X, y):
            self.le_ = LabelEncoder().fit(y); self.classes_ = self.le_.classes_
            self.m_ = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                    learning_rate=self.learning_rate, random_state=SEED, n_jobs=-1)
            self.m_.fit(X, self.le_.transform(y)); return self

        def predict_proba(self, X):
            return self.m_.predict_proba(X)

        def predict(self, X):
            return self.le_.inverse_transform(self.m_.predict(X))

    EXTRA_MODELS["XGBoost"] = XGBStringLabels()
except ImportError:
    pass
try:
    from lightgbm import LGBMClassifier
    EXTRA_MODELS["LightGBM"] = LGBMClassifier(n_estimators=300, random_state=SEED, verbose=-1)
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier
    EXTRA_MODELS["CatBoost"] = CatBoostClassifier(iterations=400, random_seed=SEED, verbose=0)
except ImportError:
    pass


def all_models():
    return {**BASELINE_MODELS, **EXTRA_MODELS}
