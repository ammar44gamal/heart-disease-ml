import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# 1. Load cleaned dataset (before PCA)
# -------------------------------------
df = pd.read_csv("heart_disease_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------------
# 2. Feature Importance using Random Forest
# -------------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()

# -------------------------------------
# 3. Recursive Feature Elimination (RFE)
# -------------------------------------
rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X, y)

selected_features_rfe = X.columns[rfe.support_]
print("\n📌 Top 5 features selected by RFE:")
print(selected_features_rfe)

# -------------------------------------
# 4. Chi-Square Test (only works with non-negative values)
# -------------------------------------
# First scale values between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply chi2 test
chi_selector = SelectKBest(score_func=chi2, k=5)
chi_selector.fit(X_scaled, y)
selected_features_chi2 = X.columns[chi_selector.get_support()]

print("\n📌 Top 5 features selected by Chi-Square Test:")
print(selected_features_chi2)

# -------------------------------------
# 5. Save reduced dataset (based on RFE)
# -------------------------------------
df_reduced = df[selected_features_rfe.tolist() + ['target']]
df_reduced.to_csv("heart_disease_selected_features.csv", index=False)

print("\n💾 Reduced dataset saved as 'heart_disease_selected_features.csv'")
