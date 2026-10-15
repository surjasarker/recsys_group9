# RPG — HNSW Graph Extension (RPG-HNSW)

**Original work.** Adapted from the [official RPG repository](https://github.com/facebookresearch/RPG) (Hou et al., KDD 2025). The core framework (`genrec/` excluding `genrec/models/RPG/model.py`) is unchanged from the original. The RPG model has been modified to replace the O(n²) similarity matrix with a FAISS HNSW approximate nearest neighbour index for graph construction.

## What was changed

`genrec/models/RPG/model.py` replaces `build_ii_sim_mat()` + `build_adjacency_list()` with:

- **`build_adjacency_faiss()`** — computes item embeddings as the normalized mean of their learned GPT-2 token embeddings, builds a FAISS HNSW index over all items, searches `n_edges` approximate nearest neighbours per item, and returns the adjacency tensor. Memory scales as O(n·d) instead of O(n²).

Inference (`graph_propagation()`) is identical to the original RPG beam search — only the source of the adjacency list changes.

## Why this matters

The original RPG builds an n×n item-item similarity matrix on GPU, which runs out of memory at approximately 250k items on an H100 (80 GB). RPG-HNSW builds the same adjacency structure using FAISS HNSW, requiring only ~2.5 GB CPU and ~0.4 GB GPU at 1M items — a reduction of more than three orders of magnitude.

## Running

```bash
cd RPG_KDD2025
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors
```

Available categories: `Sports_and_Outdoors`, `Beauty`, `Toys_and_Games`, `CDs_and_Vinyl`, `Movies_and_TV`.

All hyperparameters are the same as the original RPG. The HNSW index is built automatically at the start of evaluation.

## Config

See `RPG_KDD2025/genrec/models/RPG/config.yaml` for all model hyperparameters and `RPG_KDD2025/genrec/default.yaml` for training defaults.
