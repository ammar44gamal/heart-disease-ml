import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load the data
# ----------------------------
df = pd.read_csv("heart_disease_selected_features.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------
# 2. Split the data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Use your best parameters from GridSearch
# ----------------------------
best_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42
)

best_model.fit(X_train, y_train)

# ----------------------------
# 4. Save the trained model using joblib
# ----------------------------
joblib.dump(best_model, "final_model.pkl")
print("✅ Model saved as 'final_model.pkl'")

# ----------------------------
# 5. (Optional) Save the features used
# ----------------------------
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("✅ Model features saved as 'model_features.pkl'")
