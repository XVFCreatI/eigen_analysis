import numpy as np

# ─────────────────────────────────────────────────────────
# SECTION 1 — General eigenvalues & eigenvectors
# ─────────────────────────────────────────────────────────

A = np.array([
    [2.0, 7.0],
    [7.0, 2.0]
])

eigenvalues_A, eigenvectors_A = np.linalg.eig(A)

print("── General ──────────────────────")
print("Eigenvalues :", eigenvalues_A)
print("Eigenvectors:", eigenvectors_A)
print("1st vector  :", eigenvectors_A[:, 0])
print("2nd vector  :", eigenvectors_A[:, 1])


# ─────────────────────────────────────────────────────────
# SECTION 2 — PCA on sensor data
# ─────────────────────────────────────────────────────────

data = np.random.randn(200, 4)        # replace with your data
data -= data.mean(axis=0)             # centre
cov  = np.cov(data, rowvar=False)     # covariance matrix

eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(cov)

idx              = np.argsort(eigenvalues_pca)[::-1]
eigenvalues_pca  = eigenvalues_pca[idx]
eigenvectors_pca = eigenvectors_pca[:, idx]

k            = 2                      # number of components to keep
data_reduced = data @ eigenvectors_pca[:, :k]

print("\n── PCA ──────────────────────────")
print("Eigenvalues      :", eigenvalues_pca.round(4))
print("Variance explained:", (eigenvalues_pca[:k] / eigenvalues_pca.sum()).round(3))
print("Reduced data shape:", data_reduced.shape)


# ─────────────────────────────────────────────────────────
# SECTION 3 — Stability check (control system)
# ─────────────────────────────────────────────────────────

B = np.array([
    [2.0, 7.0],
    [7.0, 2.0]
])

eigenvalues_B, eigenvectors_B = np.linalg.eig(B)

is_stable = np.all(eigenvalues_B.real < 0)

print("\n── Stability ────────────────────")
print("Eigenvalues :", eigenvalues_B)
print("Real parts  :", eigenvalues_B.real)
print("Stable      :", is_stable)