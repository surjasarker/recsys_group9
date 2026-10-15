# Reproducibility Study of RPG: Retrieval via Personalized Graphs

This repository contains all code for the reproducibility paper on RPG (Hou et al., KDD 2025), including reproductions of RPG, SASRec, and TIGER, as well as three original decoding extensions: exhaustive search, cluster-based retrieval, and HNSW-based graph construction.

## Repository Structure

```
repo/
├── MemGen-GR/              # SASRec and TIGER reproduction (third-party code)
├── RPG_original/           # RPG reproduction (third-party code)
├── RPG_exhaustive search/  # Extension: RPG with exhaustive decoding (original)
├── RPG_cluster/            # Extension: RPG with cluster-based retrieval (original)
└── RPG_HNSW/               # Extension: RPG with FAISS HNSW graph (original)
```

## Attribution

**`MemGen-GR/`** is taken from the repository of Ding et al. [1] and is not original work. The SASRec and TIGER model implementations, training framework, and evaluation pipeline are from that repo. Original additions in this folder: `benchmark_latency.py` and the Slurm job scripts (`*.job`).

**`RPG_original/`** is taken from the [official RPG repository](https://github.com/facebookresearch/RPG) by Hou et al. (KDD 2025) and is not original work. Original additions in this folder: `benchmark_crossover.py` and `plot_crossover.py`.

**`RPG_exhaustive search/`**, **`RPG_cluster/`**, and **`RPG_HNSW/`** are all original extensions adapted from the RPG codebase. The core framework (`genrec/`) is inherited from RPG; the RPG model (`genrec/models/RPG/model.py`) has been modified to implement each alternative decoding strategy.

## Setup

Each subfolder has its own `requirements.txt`. The RPG-based folders share the same environment:

```bash
pip install -r RPG_original/requirements.txt
```

For MemGen-GR:

```bash
pip install -r MemGen-GR/requirements.txt
```

All experiments were run on the Snellius HPC cluster using a single NVIDIA H100 80 GB GPU.

## Datasets

All models use the Amazon Reviews 2014 dataset with 5-core filtering and leave-one-out splits. Datasets are downloaded automatically on first run. Pre-computed `text-embedding-3-large` embeddings (used by RPG) are publicly available from the original authors.

## Reproduction

See the README inside each subfolder for dataset-specific training commands.

## Extensions

| Folder | Description |
|---|---|
| `RPG_exhaustive search/` | Removes beam search; scores all items via a single dense matmul. Used as performance upper bound. |
| `RPG_cluster/` | Replaces beam search with k-means cluster pruning (IVF-style). Configurable coverage via `n_clusters` and `n_probe_clusters`. |
| `RPG_HNSW/` | Replaces the O(n²) similarity matrix with a FAISS HNSW index for graph construction. Inference is identical to original RPG beam search. |

## Latency and Memory Benchmarks

- `RPG_original/benchmark_crossover.py` — benchmarks per-query inference latency and preprocessing memory for exhaustive search, beam search, and HNSW across synthetic catalogues of 1k–1M items.
- `MemGen-GR/benchmark_latency.py` — benchmarks per-query inference latency for SASRec and TIGER on real datasets using saved checkpoints.

## References

[1] Ding et al., "On the Memorization and Generalization of Generative Recommendation", KDD 2026.

## Citation

```bibtex
@inproceedings{ding2026generalize,
  title={On the Memorization and Generalization of Generative Recommendation},
  author={Yijie Ding and Zitian Guo and Jiacheng Li and Letian Peng and Shuai Shao and Wei Shao
          and Xiaoqiang Luo and Luke Simon and Jingbo Shang and Julian McAuley and Yupeng Hou},
  booktitle={{KDD}},
  year={2026}
}

@inproceedings{hou2025rpg,
  title={Generating Long Semantic IDs in Parallel for Recommendation},
  author={Yupeng Hou and Jiacheng Li and Ashley Shin and Jinsung Jeon and
          Abhishek Santhanam and Wei Shao and Kaveh Hassani and Ning Yao and Julian McAuley},
  booktitle={{KDD}},
  year={2025}
}
```
