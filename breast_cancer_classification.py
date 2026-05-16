# Yazdan-Ghorbani
# Data Mining Project - Breast Cancer Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# Configuration

RANDOM_STATE = 42
TEST_SIZE = 0.20
SAVE_PLOT = True
PLOT_FILENAME = "roc_comparison.png"
SAVE_METRICS_CSV = True
METRICS_CSV = "model_metrics.csv"

# 1. Load and explore dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts().rename_axis('class').reset_index(name='count'))
print("\nFeature names:", list(X.columns))
print("\nFirst 5 rows:\n", X.head())

# 2. Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Preprocessing

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define models

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=10000, solver="liblinear", random_state=RANDOM_STATE),
    "SVC": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE)
}


# 5. Train, predict, evaluate

results = []
roc_curves = {}

for name, model in models.items():
    if name in ("KNN", "LogisticRegression", "SVC"):
        Xtr, Xte = X_train_scaled, X_test_scaled
    else:
        Xtr, Xte = X_train.values, X_test.values

    # Train
    model.fit(Xtr, y_train)

    # Predict labels and probabilities
    y_pred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(Xte)[:, 1]
    else:
        try:
            scores = model.decision_function(Xte)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            y_prob = y_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": roc_auc
    })
    roc_curves[name] = (fpr, tpr, roc_auc)

    print("\n" + "="*60)
    print(f"Model: {name}")
    print("- Classification report -")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
    cm = confusion_matrix(y_test, y_pred)
    print("- Confusion matrix -")
    print(cm)


# 6. Results table and save CSV

df_results = pd.DataFrame(results).sort_values(by="auc", ascending=False).reset_index(drop=True)
print("\n" + "="*60)
print("Summary metrics sorted by AUC")
print(df_results)

if SAVE_METRICS_CSV:
    df_results.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved metrics to {METRICS_CSV}")

# 7. Plot ROC curves together

plt.figure(figsize=(9, 7))
sns.set(style="whitegrid")
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", alpha=0.7)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"\nSaved ROC plot to {PLOT_FILENAME}")

plt.show()

# 8. visualize Decision Tree

plt.figure(figsize=(16,10))
plot_tree(models["DecisionTree"], feature_names=X.columns, class_names=["malignant","benign"], filled=True, max_depth=3)
plt.title("Decision Tree (first 3 levels)")
plt.show()


