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

## Reference

This project is motivated by the following paper:

> B Hell, M Kassubeck, P Bauszat (2015). *An approach toward fast gradient-based image segmentation*. … on Image Processing. https://ieeexplore.ieee.org/abstract/document/7076610/

Paper author: Prof. Martin Eisemann, Technische Universität Braunschweig (eisemann@cg.cs.tu-bs.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
