import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("heart_disease_selected_features.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------
# 2. Train/Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Train models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

# Set up the ROC curve plot
plt.figure(figsize=(10, 6))

for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"📊 Evaluation for {name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))  # avoids warnings

    # Try to get probabilities (needed for ROC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        continue  # Skip if model doesn't support predict_proba

    # Handle binary or multiclass
    if y_prob.shape[1] == 2:  # Binary
        y_scores = y_prob[:, 1]
        auc = roc_auc_score(y_test, y_scores)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
    else:  # Multiclass
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())

    print(f"✅ ROC-AUC Score: {auc:.4f}")
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

# Final plot settings
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_path = os.path.join(os.getcwd(), "roc_curve_comparison.png")
plt.savefig(save_path)
print(f"✅ ROC curve saved as: {save_path}")
plt.show()