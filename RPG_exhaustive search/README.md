# RPG — Exhaustive Search

**Original work.** Adapted from the [official RPG repository](https://github.com/facebookresearch/RPG) (Hou et al., KDD 2025). The core framework (`genrec/` excluding `genrec/models/RPG/model.py`) is unchanged from the original. The RPG model has been modified to remove graph beam search entirely, replacing it with exhaustive scoring over all catalogue items.

## What was changed

`genrec/models/RPG/model.py` removes `build_ii_sim_mat()`, `build_adjacency_list()`, and `graph_propagation()` from the original model. The `generate()` method always uses exhaustive scoring:

1. Compute token log-probabilities for the query via the GPT-2 backbone.
2. For each item, look up the log-probabilities of its semantic ID tokens and average across all codebooks.
3. Return the top-k items by score.

No preprocessing is required. Scoring is done via a direct tensor index operation, which is equivalent to a batched gather over the item token table.

## Role in the paper

Exhaustive search serves as the **performance upper bound** for all RPG decoding variants. Since it scores every item in the catalogue, it cannot be beaten by any approximate method. It also serves as the latency baseline against which cluster-based retrieval and HNSW beam search are compared.

## Running

```bash
cd RPG_KDD2025
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors
```

Available categories: `Sports_and_Outdoors`, `Beauty`, `Toys_and_Games`, `CDs_and_Vinyl`, `Movies_and_TV`.

No additional hyperparameters are needed beyond the standard RPG configuration.

## Config

See `RPG_KDD2025/genrec/models/RPG/config.yaml` for model hyperparameters and `RPG_KDD2025/genrec/default.yaml` for training defaults.
