# RPG — Cluster-Based Retrieval Extension

**Original work.** Adapted from the [official RPG repository](https://github.com/facebookresearch/RPG) (Hou et al., KDD 2025). The core framework (`genrec/` excluding `genrec/models/RPG/model.py`) is unchanged from the original. The RPG model has been modified to replace graph beam search with a cluster-based retrieval mechanism.

## What was changed

`genrec/models/RPG/model.py` adds the following to the original RPG model:

- **`_build_item_profiles()`** — constructs a binary indicator matrix representing each item as a one-hot vector over its assigned semantic ID tokens.
- **`init_clusters()`** — runs FAISS k-means over item profiles at first inference call. Stores cluster centroids, padded item-ID lists, and validity masks on GPU.
- **`cluster_search()`** — at inference, scores all cluster centroids with one matmul, gathers items from the top-p clusters, scores them, and returns the top-k.

Training is identical to the original RPG. The cluster index is built lazily on the first call to `generate()` during evaluation.

## How it works

During preprocessing, items are grouped into `n_clusters` clusters using k-means over their token-profile vectors (a binary representation of which codebook tokens each item uses). At inference, the model:

1. Scores all cluster centroids against the query using a single matmul.
2. Selects the top `n_probe_clusters` clusters.
3. Exhaustively scores only the items within those clusters.

This reduces the number of items scored from N to approximately `N * (n_probe_clusters / n_clusters)`, controlling the coverage–quality trade-off.

## Running

```bash
cd RPG_KDD2025
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --n_clusters=500 \
    --n_probe_clusters=20
```

Available categories: `Sports_and_Outdoors`, `Beauty`, `Toys_and_Games`, `CDs_and_Vinyl`, `Movies_and_TV`.

Key hyperparameters (in addition to base RPG hyperparameters):

| Parameter | Description |
|---|---|
| `n_clusters` | Number of k-means clusters |
| `n_probe_clusters` | Number of clusters searched per query |

## Config

See `RPG_KDD2025/genrec/models/RPG/config.yaml` for all model hyperparameters and `RPG_KDD2025/genrec/default.yaml` for training defaults.
