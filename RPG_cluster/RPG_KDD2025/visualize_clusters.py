"""
2D scatter plot of RPG item clusters via PCA -> t-SNE.
Run from the RPG_cluster directory:

  python visualize_clusters.py --category Toys_and_Games --n_clusters 300
  python visualize_clusters.py --category Movies_and_TV --n_clusters 300 --sample 5000
"""
import argparse
import os
import numpy as np
import torch
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from accelerate import Accelerator

from genrec.utils import (
    get_config, init_seed, init_logger, init_device,
    get_dataset, get_tokenizer
)


def build_profiles(item_id2tokens: torch.Tensor, n_digit: int, codebook_size: int) -> np.ndarray:
    total_tokens = n_digit * codebook_size
    n_real = item_id2tokens.shape[0] - 1
    flat_ids = item_id2tokens[1:].numpy() - 1  # 0-indexed flat token positions
    profiles = np.zeros((n_real, total_tokens), dtype=np.float32)
    profiles[np.arange(n_real)[:, None], flat_ids] = 1.0
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', required=True, help='AmazonReviews2014 category')
    parser.add_argument('--n_clusters', type=int, default=300)
    parser.add_argument('--pca_dims', type=int, default=50,
                        help='Intermediate PCA dimensionality before t-SNE')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly subsample N items before t-SNE (recommended for >20k items)')
    parser.add_argument('--output', type=str, default=None, help='Output PNG path')
    args = parser.parse_args()

    # ── Config + infrastructure ──────────────────────────────────────────────
    config = get_config(
        model_name='RPG',
        dataset_name='AmazonReviews2014',
        config_file=None,
        config_dict={'category': args.category}
    )
    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()
    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)

    # ── Dataset + tokenizer (loads from cache — no API calls or retraining) ──
    dataset = get_dataset('AmazonReviews2014')(config)
    dataset.split()  # populates dataset.split_data, required by tokenizer._get_items_for_training
    tokenizer = get_tokenizer('RPG')(config, dataset)

    # ── Reconstruct item_id2tokens (same logic as RPG.__init__) ──────────────
    item_id2tokens = torch.zeros((dataset.n_items, tokenizer.n_digit), dtype=torch.long)
    for item, tokens in tokenizer.item2tokens.items():
        item_id2tokens[dataset.item2id[item]] = torch.LongTensor(tokens)

    n_digit = tokenizer.n_digit
    codebook_size = tokenizer.codebook_size
    n_real = dataset.n_items - 1
    print(f"\n[VIZ] {args.category}: {n_real} items | profile dim = {n_digit}x{codebook_size} = {n_digit*codebook_size}")

    # ── Build binary item profiles ───────────────────────────────────────────
    profiles = build_profiles(item_id2tokens, n_digit, codebook_size)

    # ── k-means ──────────────────────────────────────────────────────────────
    print(f"[VIZ] k-means: k={args.n_clusters}, niter=20 ...")
    km = faiss.Kmeans(
        d=profiles.shape[1],
        k=args.n_clusters,
        niter=20,
        seed=int(config['rand_seed']),
        verbose=False
    )
    km.train(profiles)
    _, asgn = km.index.search(profiles, 1)
    asgn = asgn.squeeze(1)  # (n_real,)

    # ── Subsample ────────────────────────────────────────────────────────────
    if args.sample is not None and args.sample < n_real:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_real, args.sample, replace=False)
        profiles_vis = profiles[idx]
        labels_vis = asgn[idx]
        n_vis = args.sample
        print(f"[VIZ] Subsampled {n_vis}/{n_real} items for t-SNE")
    else:
        profiles_vis = profiles
        labels_vis = asgn
        n_vis = n_real

    # ── PCA ──────────────────────────────────────────────────────────────────
    print(f"[VIZ] PCA: {profiles_vis.shape[1]} -> {args.pca_dims} dims ...")
    pca = PCA(n_components=args.pca_dims, random_state=42)
    emb_pca = pca.fit_transform(profiles_vis)
    print(f"[VIZ] PCA cumulative explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"[VIZ] t-SNE: {args.pca_dims} -> 2, perplexity={args.perplexity} ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=1000, random_state=42, verbose=1)
    coords = tsne.fit_transform(emb_pca)

    # ── Plot ─────────────────────────────────────────────────────────────────
    out = args.output or f"clusters_{args.category}_{args.n_clusters}.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels_vis,
        cmap='nipy_spectral',
        s=4, alpha=0.6, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label='Cluster ID')
    subtitle = f"{n_vis}/{n_real} items sampled" if args.sample else f"{n_real} items"
    ax.set_title(
        f'{args.category}  —  {args.n_clusters} clusters  ({subtitle})\n'
        f'PCA({args.pca_dims}) → t-SNE(perplexity={args.perplexity})'
    )
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[VIZ] Saved → {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
