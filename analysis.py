"""Baseline: per-pixel k-means colour segmentation (RGB, no spatial term).
Improved: same k-means colour centroids and data costs, plus a convex total-variation
smoothness term on the per-pixel label simplex, minimised by projected gradient descent.
Seeds 42, 43, 44 are used for the randomised k-means; the test set is used once at the end."""
import json, os, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, SEEDS = 42, [42, 43, 44]
SIZES, KS = [64, 96], [3, 5]
N_VAL, N_TEST = 10, 30            # images per configuration
LAMBDAS = [0.1, 0.3, 1.0, 3.0]    # smoothness grid, chosen on validation only
PGD_ITERS = 300                   # projected gradient iterations
EPS = 0.1                         # smoothing of |.| in the TV term
NOISE = 0.08                      # Gaussian colour noise std


def make_image(size, k, seed):
    """Random Voronoi cells, each labelled with one of k well-separated colours, plus noise."""
    rng = np.random.default_rng(seed)
    cols = []
    while len(cols) < k:  # rejection-sample colours that are at least 0.25 apart
        c = rng.uniform(0, 1, 3)
        if all(np.linalg.norm(c - d) > 0.25 for d in cols):
            cols.append(c)
    cols = np.array(cols)
    n_cells = 3 * k
    seeds_xy = rng.uniform(0, size, (n_cells, 2))
    cell_lab = rng.permutation(np.arange(n_cells) % k)
    yy, xx = np.mgrid[0:size, 0:size]
    pix = np.stack([yy.ravel(), xx.ravel()], 1).astype(float)
    cell = np.argmin(((pix[:, None, :] - seeds_xy[None]) ** 2).sum(-1), axis=1)
    gt = cell_lab[cell].reshape(size, size)
    img = cols[gt] + rng.normal(0, NOISE, (size, size, 3))
    return np.clip(img, 0, 1), gt


def build(n_per, seed_offset):
    """Build all (size, k) configurations; seed offsets keep train and test images disjoint."""
    items = []
    for si, s in enumerate(SIZES):
        for ki, k in enumerate(KS):
            for i in range(n_per):
                img, gt = make_image(s, k, seed_offset + 1000 * si + 100 * ki + i)
                items.append((s, k, img, gt))
    return items


def project_simplex(V):
    """Row-wise Euclidean projection onto the probability simplex."""
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1)
    ind = np.arange(1, V.shape[1] + 1)
    cond = U - (css - 1) / ind > 0
    rho = V.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = (css[np.arange(V.shape[0]), rho] - 1) / (rho + 1)
    return np.maximum(V - theta[:, None], 0)


def tv_grad(u, lam):
    """Gradient of lam * sum over 4-neighbour edges of sqrt(diff^2 + EPS^2) (smoothed L1 TV)."""
    g = np.zeros_like(u)
    for ax in (0, 1):
        diff = np.diff(u, axis=ax)
        phi = diff / np.sqrt(diff ** 2 + EPS ** 2)
        if ax == 0:
            g[1:] += phi; g[:-1] -= phi
        else:
            g[:, 1:] += phi; g[:, :-1] -= phi
    return lam * g


def pgd_refine(img, labels, centers, lam):
    """Minimise sum(D*u) + TV(u) over the simplex with projected gradient descent."""
    H, W, _ = img.shape
    K = len(centers)
    d2 = ((img.reshape(-1, 1, 3) - centers[None]) ** 2).sum(-1)
    D = (d2 / d2.mean()).reshape(H, W, K)          # data cost, scaled to unit mean
    u = np.eye(K)[labels].reshape(H, W, K)         # warm start at k-means assignment
    tau = EPS / (8 * lam)                          # 1 / Lipschitz bound of the TV gradient
    for _ in range(PGD_ITERS):
        u = project_simplex((u - tau * (D + tv_grad(u, lam))).reshape(-1, K)).reshape(H, W, K)
    return np.argmax(u, axis=2)


def evaluate(gt, pred, k):
    """ARI against ground truth and pixel accuracy after Hungarian label matching."""
    ari = adjusted_rand_score(gt.ravel(), pred.ravel())
    C = np.zeros((k, k))
    np.add.at(C, (gt.ravel(), pred.ravel()), 1)
    r, c = linear_sum_assignment(-C)
    return ari, C[r, c].sum() / gt.size


def run_method(name, items, seed, lam):
    """Run one model over all images; returns {(size,k): [(ari, acc, seconds), ...]}."""
    res = {}
    for s, k, img, gt in items:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(img.reshape(-1, 3))
        if name == "baseline":
            pred = km.labels_.reshape(s, s)
        else:
            pred = pgd_refine(img, km.labels_, km.cluster_centers_, lam)
        dt = time.perf_counter() - t0
        ari, acc = evaluate(gt, pred, k)
        res.setdefault((s, k), []).append((ari, acc, dt))
    return res


def summarise(res):
    """Mean ARI, pixel accuracy and seconds per image over all configurations."""
    v = np.array([x for lst in res.values() for x in lst])
    return v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()


def cfg_mean(runs, cfg, idx):
    return float(np.mean([x[idx] for r in runs for x in r[cfg]]))


def main():
    os.makedirs("figures", exist_ok=True)
    print("Generating synthetic images...")
    val, test = build(N_VAL, 0), build(N_TEST, 100000)
    print(f"validation images={len(val)}, test images={len(test)}")

    # Select the smoothness weight on validation images only (seed 42)
    val_scores = {}
    for lam in LAMBDAS:
        val_scores[lam] = summarise(run_method("improved", val, SEED, lam))[0]
        print(f"  lambda={lam}: validation ARI={val_scores[lam]:.4f}")
    best = max(LAMBDAS, key=lambda l: val_scores[l])
    print(f"Selected lambda={best}")

    # Final test evaluation, once per seed
    base_runs, imp_runs = [], []
    for sd in SEEDS:
        print(f"Test evaluation, seed={sd}")
        base_runs.append(run_method("baseline", test, sd, None))
        imp_runs.append(run_method("improved", test, sd, best))
    b = np.array([summarise(r) for r in base_runs])
    m = np.array([summarise(r) for r in imp_runs])

    metrics = {
        "baseline_score": float(b[:, 0].mean()),
        "improved_score": float(m[:, 0].mean()),
        "baseline_score_std": float(b[:, 0].std()),
        "improved_score_std": float(m[:, 0].std()),
        "improvement": float(m[:, 0].mean() - b[:, 0].mean()),
        "baseline_pixel_acc": float(b[:, 1].mean()),
        "improved_pixel_acc": float(m[:, 1].mean()),
        "baseline_runtime_s": float(b[:, 2].mean()),
        "improved_runtime_s": float(m[:, 2].mean()),
        "lambda_selected": float(best),
        "validation_ari_selected": float(val_scores[best]),
    }
    cfgs = [(s, k) for s in SIZES for k in KS]
    for s, k in cfgs:
        t = f"{s}px_k{k}"
        metrics[f"baseline_ari_{t}"] = cfg_mean(base_runs, (s, k), 0)
        metrics[f"improved_ari_{t}"] = cfg_mean(imp_runs, (s, k), 0)
        metrics[f"baseline_runtime_s_{t}"] = cfg_mean(base_runs, (s, k), 2)
        metrics[f"improved_runtime_s_{t}"] = cfg_mean(imp_runs, (s, k), 2)
    if not all(np.isfinite(v) for v in metrics.values()):
        print("Non-finite metric produced; exiting.")
        sys.exit(1)

    # Charts: ARI per configuration and runtime per image
    tags = [f"{s}px\nk={k}" for s, k in cfgs]
    x = np.arange(len(cfgs))
    for fname, key, ylab in [("figures/ari_by_config.png", "ari", "ARI (test)"),
                             ("figures/runtime_by_config.png", "runtime_s", "seconds per image")]:
        plt.figure(figsize=(7, 4))
        plt.bar(x - 0.2, [metrics[f"baseline_{key}_{s}px_k{k}"] for s, k in cfgs], 0.4, label="baseline (k-means)")
        plt.bar(x + 0.2, [metrics[f"improved_{key}_{s}px_k{k}"] for s, k in cfgs], 0.4, label="improved (TV-PGD)")
        plt.xticks(x, tags); plt.ylabel(ylab); plt.legend(); plt.tight_layout()
        plt.savefig(fname, dpi=100); plt.close()

    results = {
        "question": "Does a convex total-variation multilabel relaxation solved by projected gradient descent give better region agreement with ground truth than a standard per-pixel k-means colour segmentation, and at what runtime cost as the number of labels and image size grow?",
        "method": "Seeded synthetic Voronoi colour images; baseline k-means on RGB vs k-means colours plus convex TV smoothing on the label simplex solved by projected gradient descent; lambda chosen on validation images; test evaluated once per seed (42, 43, 44).",
        "dataset": {
            "name": "Seeded synthetic Voronoi colour images (generated in code)",
            "source_url": "https://numpy.org/doc/stable/reference/random/generator.html",
            "n_rows": len(val) + len(test),
            "n_train": len(val),
            "n_test": len(test),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "adjusted Rand index (ARI) vs ground truth, mean over test images, averaged over 3 seeds",
        "higher_is_better": True,
        "baseline": "Per-pixel k-means colour clustering on RGB (scikit-learn KMeans, n_init=10), no spatial term.",
        "improved": "Same k-means colour centroids and data costs, plus a convex smoothed total-variation term over 4-neighbour pixel pairs on the label simplex, minimised by projected gradient descent (300 iterations, warm start at k-means labels).",
        "metrics": metrics,
        "notes": "Synthetic data only, so results do not reproduce the paper's dataset or reported numbers. Only 2 image sizes and 2 label counts were tested, so scaling is indicative. Validation images were generated separately from test images and used only to pick lambda. Seeds affect only k-means initialisation, so the std reflects that. Runtime is wall-clock on the CI machine and includes the shared k-means step.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")
    print(f"baseline ARI={metrics['baseline_score']:.4f}  improved ARI={metrics['improved_score']:.4f}")


if __name__ == "__main__":
    main()
