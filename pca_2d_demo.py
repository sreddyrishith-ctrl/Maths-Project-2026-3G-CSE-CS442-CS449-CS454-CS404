"""
pca_2d_demo.py  --  PCA on 5 points in 2D, small enough to verify by hand.

Shows, step by step:
  1. centre the data
  2. covariance matrix
  3. eigenvalues / eigenvectors (the principal axes)
  4. projection of every point onto the first principal axis
  5. a check that "projection matrix P = u u^T" gives the same answer

Run:  python pca_2d_demo.py
Saves: fig_pca_2d.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ---- the data: 5 points, each row is (x, y) --------------------------------
X = np.array([[2.0, 1.0],
              [3.0, 4.0],
              [5.0, 3.0],
              [7.0, 7.0],
              [8.0, 5.0]])
N = len(X)

# ---- 1. centre ---------------------------------------------------------------
mean = X.mean(axis=0)
A = X - mean
print("1. mean =", mean)
print("   centred data A =\n", A, "\n")

# ---- 2. covariance matrix  C = A^T A / (N-1) --------------------------------
C = A.T @ A / (N - 1)
print("2. covariance matrix C =\n", C, "\n")

# ---- 3. eigen-decomposition (eigh: C is symmetric -> real, orthogonal vecs) --
vals, vecs = np.linalg.eigh(C)
order = np.argsort(vals)[::-1]            # largest variance first
vals, vecs = vals[order], vecs[:, order]
for i in range(2):
    # check the definition: C v = lambda v
    print(f"3. lambda_{i+1} = {vals[i]:.4f}, v_{i+1} = {vecs[:, i]},"
          f"  check C v - lambda v = {C @ vecs[:, i] - vals[i] * vecs[:, i]}")
print(f"   v1 . v2 = {vecs[:, 0] @ vecs[:, 1]:.1e}  (orthogonal, as expected for a symmetric matrix)")
print(f"   variance explained by PC1 = {vals[0] / vals.sum():.1%}\n")

# ---- 4. project onto PC1 -------------------------------------------------------
u = vecs[:, 0]                             # unit vector along PC1
scores = A @ u                             # 1-number description of each point
proj = np.outer(scores, u) + mean          # where those points land in 2D
print("4. PC1 scores (1D coordinates) =", scores)
print("   projected points (back in 2D) =\n", proj, "\n")

# ---- 5. same thing using the projection matrix P = u u^T ----------------------
P = np.outer(u, u)
proj2 = A @ P + mean
print("5. P = u u^T =\n", P)
print("   max difference between the two methods:", np.abs(proj - proj2).max())
print("   P @ P == P (projecting twice changes nothing):", np.allclose(P @ P, P))

err = ((X - proj) ** 2).sum()
print(f"   total squared error lost by dropping PC2 = {err:.4f}"
      f"  (= (N-1) * lambda_2 = {(N-1) * vals[1]:.4f})")

# ---- plot ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(*X.T, s=70, color="tab:blue", zorder=3, label="original points")
ax.scatter(*proj.T, s=50, color="tab:red", zorder=3, label="projections onto PC1")
for p, q in zip(X, proj):                  # dotted lines = the shortest distance
    ax.plot([p[0], q[0]], [p[1], q[1]], "k:", lw=1)
ax.scatter(*mean, marker="x", s=100, color="k", zorder=4, label="mean")

t = np.linspace(-6, 6, 2)
ax.plot(*(mean[:, None] + np.outer(vecs[:, 0], t)), color="tab:red", lw=1.5, alpha=.6)
for i, c in enumerate(["tab:red", "tab:green"]):
    d = vecs[:, i] * np.sqrt(vals[i]) * 2          # arrow length ~ 2 std devs
    ax.annotate("", xy=mean + d, xytext=mean,
                arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
    ax.text(*(mean + d * 1.08), f"PC{i+1}", color=c, fontsize=12, weight="bold")

ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper left")
ax.set_title("PCA in 2D: projection onto the direction of maximum variance")
fig.tight_layout(); fig.savefig("fig_pca_2d.png", dpi=150)
print("\nSaved fig_pca_2d.png")
