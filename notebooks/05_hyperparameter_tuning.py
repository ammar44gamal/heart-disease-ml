import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("heart_disease_selected_features.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 2. Define base model
# ----------------------------
rf = RandomForestClassifier(random_state=42)

# ----------------------------
# 3. Define hyperparameter grid
# ----------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ----------------------------
# 4. Grid Search (exhaustive search)
# ----------------------------
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

print("🔍 Running GridSearchCV...")
grid_search.fit(X_train, y_train)

# Best result
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import label_binarize

# Binarize y_test for multiclass ROC-AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
roc_auc = roc_auc_score(y_test_bin, best_model.predict_proba(X_test), multi_class='ovr')

print(f"ROC-AUC Score (multi-class): {roc_auc:.4f}")