"""
eigenfaces.py  --  Face recognition with PCA ("eigenfaces"), written from scratch in NumPy.

Pipeline
  1. Load faces; every image (64x64) is flattened into a vector in R^4096
  2. Split: 8 images per person for training (the "gallery"), 2 for testing
  3. Mean face -> centre the data
  4. PCA via the small N x N matrix trick  (and verified against SVD)
  5. Eigenfaces = the eigenvectors reshaped back into images
  6. Project faces into "face space" (a short vector of k numbers per face)
  7. Recognise a test face = nearest neighbour in face space
  8. Experiments: reconstruction, accuracy vs k, rejecting unknown people

Run:
    python eigenfaces.py                    # real Olivetti faces (downloads ~1.4 MB once)
    python eigenfaces.py --data synthetic   # fake data, only to test the code offline

Needs: numpy, matplotlib, scikit-learn (scikit-learn is used ONLY to download the dataset;
all the PCA maths below is plain NumPy).
Saves figures into ./eigenface_figures/
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIDE = 64                      # images are SIDE x SIDE
rng = np.random.default_rng(0)


# =============================================================================
# 1. Data
# =============================================================================
def load_data(source):
    """Return images (n, 64, 64) in [0,1] and integer person labels (n,)."""
    if source == "olivetti":
        from sklearn.datasets import fetch_olivetti_faces
        d = fetch_olivetti_faces(shuffle=False)
        return d.images.astype(float), d.target
    # synthetic: 40 "people", each a base pattern of smooth blobs + random lighting + noise.
    yy, xx = np.mgrid[0:SIDE, 0:SIDE] / SIDE
    imgs, labels = [], []
    for person in range(40):
        base = np.zeros((SIDE, SIDE))
        for _ in range(6):
            cx, cy, s = rng.uniform(.2, .8), rng.uniform(.2, .8), rng.uniform(.05, .15)
            base += rng.uniform(.3, 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
        for _ in range(10):
            light = rng.uniform(-.3, .3) * xx + rng.uniform(-.3, .3) * yy
            imgs.append(np.clip(base / base.max() + light + .03 * rng.standard_normal((SIDE, SIDE)), 0, 1))
            labels.append(person)
    return np.array(imgs), np.array(labels)


def split_per_person(labels, n_train=8):
    """For each person: n_train random images -> train, the rest -> test."""
    train_idx, test_idx = [], []
    for p in np.unique(labels):
        idx = rng.permutation(np.where(labels == p)[0])
        train_idx += list(idx[:n_train]); test_idx += list(idx[n_train:])
    return np.array(train_idx), np.array(test_idx)


# =============================================================================
# 2. PCA from scratch
# =============================================================================
def fit_pca(X):
    """
    X: (N, d) one flattened face per row, with N << d  (e.g. 320 faces, 4096 pixels).

    Covariance matrix would be d x d = 4096 x 4096 -- big. Trick:
        A = centred data (N x d).   C = A^T A / (N-1)        (d x d, big)
        Let L = A A^T               (N x N, small)
        If L v = lambda v   then   A^T A (A^T v) = lambda (A^T v)
        => u = A^T v is an eigenvector of A^T A with the SAME eigenvalue.
    So: diagonalise the small L, then map back with A^T and normalise.

    Returns mean (d,), eigenfaces U (d, m) with unit-length columns, variances (m,)
    sorted from largest to smallest.
    """
    N = X.shape[0]
    mean = X.mean(axis=0)
    A = X - mean
    L = A @ A.T
    vals, vecs = np.linalg.eigh(L)                 # ascending order
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-9 * vals[0]                   # drop the zero eigenvalues (rank <= N-1)
    vals, vecs = vals[keep], vecs[:, keep]
    U = A.T @ vecs                                 # map back to pixel space
    U /= np.linalg.norm(U, axis=0)                 # make each eigenface a unit vector
    return mean, U, vals / (N - 1)


def project(X, mean, U, k):
    """Face(s) -> k coordinates in face space."""
    return (X - mean) @ U[:, :k]


def reconstruct(W, mean, U):
    """k coordinates -> approximate face (mean face + weighted sum of eigenfaces)."""
    return mean + W @ U[:, :W.shape[1]].T


# =============================================================================
# 3. Recognition
# =============================================================================
def nearest_neighbour(gallery, queries):
    """Index of, and Euclidean distance to, the closest gallery row for every query row."""
    d2 = ((queries ** 2).sum(1)[:, None] + (gallery ** 2).sum(1)[None, :]
          - 2 * queries @ gallery.T)
    d2 = np.maximum(d2, 0)
    idx = d2.argmin(1)
    return idx, np.sqrt(d2[np.arange(len(queries)), idx])


# =============================================================================
# helpers for plotting
# =============================================================================
def show(ax, vec, title=None, color="k"):
    img = vec.reshape(SIDE, SIDE)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)       # stretch to [0,1] for display
    ax.imshow(img, cmap="gray"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, color=color)


def main(source, outdir):
    os.makedirs(outdir, exist_ok=True)
    save = lambda fig, name: (fig.tight_layout(), fig.savefig(os.path.join(outdir, name), dpi=150),
                              plt.close(fig))

    images, labels = load_data(source)
    n_people = len(np.unique(labels))
    flat = images.reshape(len(images), -1)
    tr, te = split_per_person(labels)
    Xtr, ytr, Xte, yte = flat[tr], labels[tr], flat[te], labels[te]
    print(f"{len(images)} images of {n_people} people | train {len(tr)} | test {len(te)} "
          f"| each image = {flat.shape[1]} numbers")

    # ------------------------------------------------------------------ fit PCA
    mean, U, var = fit_pca(Xtr)
    m = U.shape[1]
    print(f"PCA found {m} non-zero components (rank of centred training data <= N-1 = {len(tr)-1})")

    # sanity check: same eigenvalues / eigenvectors as SVD?
    _, s, Vt = np.linalg.svd(Xtr - mean, full_matrices=False)
    print("check vs SVD: max |eigenvalue difference| =",
          f"{np.abs(var - s[:m] ** 2 / (len(tr) - 1)).max():.2e}")
    print("check vs SVD: |cos angle| of first 5 eigenfaces =",
          np.round(np.abs((U[:, :5] * Vt[:5].T).sum(0)), 6), "(1 means same direction; sign is arbitrary)")
    print("check: eigenfaces orthonormal ->", np.allclose(U.T @ U, np.eye(m), atol=1e-6))

    # ------------------------------------------------ fig 1: mean face + eigenfaces
    fig, axes = plt.subplots(3, 6, figsize=(11, 7.5))
    show(axes[0, 0], mean, "mean face")
    for i, ax in enumerate(axes.ravel()[1:]):
        show(ax, U[:, i], f"eigenface {i+1}\n(var {var[i]:.2f})")
    for ax in axes.ravel()[1 + 17:]:
        ax.axis("off")
    fig.suptitle("Mean face and the top eigenfaces")
    save(fig, "fig1_mean_and_eigenfaces.png")

    # ------------------------------------------------ fig 2: variance explained
    ratio = var / var.sum(); cum = np.cumsum(ratio)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(np.arange(1, m + 1), var); a1.set_yscale("log")
    a1.set_xlabel("component"); a1.set_ylabel("eigenvalue (variance), log scale"); a1.set_title("Eigenvalue spectrum")
    a2.plot(np.arange(1, m + 1), cum)
    for target in (.9, .95, .99):
        k_t = int(np.searchsorted(cum, target) + 1)
        a2.axhline(target, color="gray", ls=":"); a2.text(m * .55, target - .035, f"{target:.0%} -> k = {k_t}")
        print(f"{target:.0%} of the variance needs k = {k_t} components (out of {flat.shape[1]} pixels)")
    a2.set_xlabel("number of components k"); a2.set_ylabel("cumulative variance explained")
    a2.set_title("How many eigenfaces do we need?")
    save(fig, "fig2_variance_explained.png")

    # ------------------------------------------------ fig 3: reconstruction
    ks = [1, 5, 10, 25, 50, 100, 200]
    ks = [k for k in ks if k <= m]
    face = Xte[0]
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(2 * (len(ks) + 1), 2.6))
    show(axes[0], face, "original")
    for ax, k in zip(axes[1:], ks):
        rec = reconstruct(project(face[None], mean, U, k), mean, U)[0]
        show(ax, rec, f"k = {k}\nerr {np.linalg.norm(face - rec):.2f}")
    fig.suptitle("Reconstructing a test face from k numbers", y=1.02)
    save(fig, "fig3_reconstruction.png")

    # ------------------------------------------------ experiment: accuracy vs k
    k_list = [k for k in [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, m] if k <= m]
    k_list = sorted(set(k_list))
    accs = []
    for k in k_list:
        gi, _ = nearest_neighbour(project(Xtr, mean, U, k), project(Xte, mean, U, k))
        accs.append((ytr[gi] == yte).mean())
    gi, _ = nearest_neighbour(Xtr, Xte)                        # baseline: raw pixels, no PCA
    base = (ytr[gi] == yte).mean()
    print("\n  k | recognition accuracy")
    for k, a in zip(k_list, accs):
        print(f"{k:3d} | {a:.1%}")
    print(f"raw {flat.shape[1]}-pixel nearest neighbour (no PCA): {base:.1%}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(k_list, accs, "o-", label="eigenfaces (k components)")
    ax.axhline(base, color="gray", ls="--", label=f"raw pixels ({flat.shape[1]} numbers)")
    ax.set_xscale("log"); ax.set_xlabel("k (number of eigenfaces used)"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(); ax.set_title("Recognition accuracy vs k")
    save(fig, "fig4_accuracy_vs_k.png")

    # ------------------------------------------------ fig 5: example matches
    k = min(50, m)
    gi, dist = nearest_neighbour(project(Xtr, mean, U, k), project(Xte, mean, U, k))
    pick = np.concatenate([np.where(ytr[gi] == yte)[0][:5], np.where(ytr[gi] != yte)[0][:3]])[:8]
    fig, axes = plt.subplots(2, len(pick), figsize=(1.8 * len(pick), 4.2), squeeze=False)
    for j, i in enumerate(pick):
        ok = ytr[gi[i]] == yte[i]
        show(axes[0, j], Xte[i], f"test: person {yte[i]}")
        show(axes[1, j], Xtr[gi[i]], f"match: person {ytr[gi[i]]}\n{'correct' if ok else 'WRONG'}",
             "green" if ok else "red")
    fig.suptitle(f"Test faces (top) and the closest stored face in face space, k = {k} (bottom)")
    save(fig, "fig5_example_matches.png")

    # ------------------------------------------------ experiment: unknown people
    # Re-fit PCA using ONLY the first ~85% of people. The remaining people are "strangers":
    # the system has never seen them, so it should say "unknown".
    n_known = int(n_people * .85)
    known = np.where(labels < n_known)[0]
    in_tr, in_te = np.isin(tr, known), np.isin(te, known)
    Xg, yg = flat[tr[in_tr]], labels[tr[in_tr]]          # gallery (known people, training images)
    Xk, yk = flat[te[in_te]], labels[te[in_te]]          # known people, held-out images
    Xu = flat[np.where(labels >= n_known)[0]]            # all images of strangers
    mean2, U2, _ = fit_pca(Xg)
    k = min(50, U2.shape[1])
    G = project(Xg, mean2, U2, k)
    gi, d_known = nearest_neighbour(G, project(Xk, mean2, U2, k))
    _, d_unk = nearest_neighbour(G, project(Xu, mean2, U2, k))
    id_acc = (yg[gi] == yk).mean()
    cands = np.sort(np.concatenate([d_known, d_unk]))
    scores = [(((d_known <= t) & (yg[gi] == yk)).sum() + (d_unk > t).sum()) / (len(d_known) + len(d_unk))
              for t in cands]
    t_best = cands[int(np.argmax(scores))]
    print(f"\nUnknown-person experiment (k={k}): {n_known} known people, {n_people - n_known} strangers")
    print(f"  identification accuracy on known people (no rejection): {id_acc:.1%}")
    print(f"  best distance threshold = {t_best:.2f}  -> overall correct decisions = {max(scores):.1%}")
    print("  (threshold was tuned on the test set, so this number is optimistic -- say so in the report)")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, max(d_known.max(), d_unk.max()), 30)
    ax.hist(d_known, bins, alpha=.6, label="known people (held-out images)")
    ax.hist(d_unk, bins, alpha=.6, label="strangers")
    ax.axvline(t_best, color="k", ls="--", label="threshold")
    ax.set_xlabel("distance to nearest stored face (in face space)"); ax.set_ylabel("count")
    ax.set_title("Rejecting unknown faces with a distance threshold"); ax.legend()
    save(fig, "fig6_unknown_threshold.png")

    print(f"\nFigures saved in ./{outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["olivetti", "synthetic"], default="olivetti")
    ap.add_argument("--out", default="eigenface_figures")
    a = ap.parse_args()
    main(a.data, a.out)
