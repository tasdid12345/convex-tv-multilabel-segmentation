# Convex TV Multilabel Relaxation vs k-means Colour Segmentation

Motivation: "An approach toward fast gradient-based image segmentation" (2015) proposes gradient-based convex optimisation for multilabel colour image segmentation. This repository tests a simple version of that idea against a standard baseline, to see whether spatial smoothing helps and what it costs in runtime.

Question: Does a convex total-variation multilabel relaxation solved by projected gradient descent give better region agreement with ground truth than a standard per-pixel k-means colour segmentation, and at what runtime cost as the number of labels and image size grow?

Dataset: seeded synthetic colour images (64x64 and 96x96 pixels, with 3 or 5 regions built from random Voronoi cells, Gaussian colour noise, and known ground-truth labels). They are generated in code with no download. This is not the paper's dataset, and the paper's dataset is not stated.

Baseline: per-pixel k-means on RGB colour, with no spatial term.

Improved model: the same k-means centroids and data costs, plus a convex total-variation smoothness term on the per-pixel label simplex, minimised by projected gradient descent. The smoothness weight is chosen on a separate validation set.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and charts to figures/).

## Results (latest run)

On seeded synthetic Voronoi colour images (120 test images, 3 seeds), per-pixel k-means colour segmentation reached 0.974 adjusted Rand index against ground truth, while my convex TV multilabel relaxation reached 0.999, at 0.31 s versus 0.015 s per image.

| Metric | Value |
|---|---|
| baseline_score | 0.9744 |
| improved_score | 0.9994 |
| baseline_score_std | 0.000001179 |
| improved_score_std | 0.000001032 |
| improvement | 0.02507 |
| baseline_pixel_acc | 0.9901 |
| improved_pixel_acc | 0.9998 |
| baseline_runtime_s | 0.01472 |
| improved_runtime_s | 0.3077 |
| lambda_selected | 0.1 |
| validation_ari_selected | 0.9995 |
| baseline_ari_64px_k3 | 0.9807 |
| improved_ari_64px_k3 | 0.9995 |
| baseline_runtime_s_64px_k3 | 0.009394 |
| improved_runtime_s_64px_k3 | 0.1679 |
| baseline_ari_64px_k5 | 0.9724 |
| improved_ari_64px_k5 | 0.9991 |
| baseline_runtime_s_64px_k5 | 0.01397 |
| improved_runtime_s_64px_k5 | 0.2311 |
| baseline_ari_96px_k3 | 0.9712 |
| improved_ari_96px_k3 | 0.9995 |
| baseline_runtime_s_96px_k3 | 0.01464 |
| improved_runtime_s_96px_k3 | 0.3523 |
| baseline_ari_96px_k5 | 0.9732 |
| improved_ari_96px_k5 | 0.9996 |
| baseline_runtime_s_96px_k5 | 0.02088 |
| improved_runtime_s_96px_k5 | 0.4795 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data only, so results do not reproduce the paper's dataset or reported numbers. Only 2 image sizes and 2 label counts were tested, so scaling is indicative. Validation images were generated separately from test images and used only to pick lambda. Seeds affect only k-means initialisation, so the std reflects that. Runtime is wall-clock on the CI machine and includes the shared k-means step.

## Reference

This project is motivated by the following paper:

> B Hell, M Kassubeck, P Bauszat (2015). *An approach toward fast gradient-based image segmentation*. … on Image Processing. https://ieeexplore.ieee.org/abstract/document/7076610/

Paper author: Prof. Martin Eisemann, Technische Universität Braunschweig (eisemann@cg.cs.tu-bs.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
