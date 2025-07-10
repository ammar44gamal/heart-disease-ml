import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------------
# 1. Load the cleaned dataset
# -------------------------------------
df = pd.read_csv("heart_disease_cleaned.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------------
# 2. Scale the features
# -------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------
# 3. Apply PCA
# -------------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# -------------------------------------
# 4. Plot explained variance ratio
# -------------------------------------
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_variance_plot.png")  # Saves the plot
plt.show()

# -------------------------------------
# 5. Keep 95% of the variance
# -------------------------------------
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"\n✅ Number of components to keep 95% variance: {pca_95.n_components_}")
print(f"New shape of dataset after PCA: {X_pca_95.shape}")

# -------------------------------------
# 6. Save the PCA-transformed data (optional)
# -------------------------------------
pca_columns = [f'PC{i+1}' for i in range(X_pca_95.shape[1])]
df_pca = pd.DataFrame(X_pca_95, columns=pca_columns)
df_pca['target'] = y.values

df_pca.to_csv("heart_disease_pca.csv", index=False)
print("\n💾 PCA-transformed dataset saved as 'heart_disease_pca.csv'")
