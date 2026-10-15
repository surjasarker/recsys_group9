"""
Crossover benchmark: exhaustive scoring vs graph beam search vs RPG-ANN.

Exhaustive:  no preprocessing, score all items at inference via chunked gather.
Beam search: one-time O(n^2) item-item similarity matrix build -> OOM at large n.
RPG-ANN:     one-time FAISS HNSW index build on mean token embeddings -> O(n*d).
             Per-query inference: identical to beam search (same adjacency format).

Run on a GPU node (no dataset, no training needed):
  python benchmark_crossover.py
"""
import time
import argparse
import resource
import numpy as np
import torch

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

# GPT-2 backbone hidden dim in the real RPG model (n_embd in config.yaml)
N_EMBD = 448


# ── Exhaustive search ─────────────────────────────────────────────────────────

def score_exhaustive(token_logits, item_id2tokens, chunk_size=4096):
    n_items = item_id2tokens.shape[0] - 1
    bs = token_logits.shape[0]
    chunks = []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        chunk_tokens = item_id2tokens[start + 1: end + 1]
        chunk_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, end - start, -1),
            dim=-1,
            index=(chunk_tokens - 1).unsqueeze(0).expand(bs, -1, -1)
        ).mean(dim=-1)
        chunks.append(chunk_logits)
    return torch.cat(chunks, dim=-1).topk(50, dim=-1).indices + 1


# ── Beam search (O(n^2) graph build) ─────────────────────────────────────────

def build_graph(n_items, n_edges, device):
    """
    Simulate build_ii_sim_mat + build_adjacency_list from RPG_original.
    Allocates the full (n_items, n_items) float32 matrix — the bottleneck
    that causes OOM at large catalogs.
    """
    sim = torch.rand((n_items, n_items), dtype=torch.float32, device=device)
    adjacency_items = torch.topk(sim, k=n_edges, dim=-1).indices
    del sim
    adjacency = torch.zeros((n_items + 1, n_edges), dtype=torch.long, device=device)
    adjacency[1:] = adjacency_items + 1
    return adjacency


def score_beam(token_logits, item_id2tokens, adjacency,
               num_beams, propagation_steps, device):
    batch_size = token_logits.shape[0]
    n_nodes = item_id2tokens.shape[0] - 1

    topk_nodes = torch.randint(1, n_nodes + 1,
                               (batch_size, num_beams),
                               dtype=torch.long, device=device)
    for _ in range(propagation_steps):
        all_neighbors = adjacency[topk_nodes].view(batch_size, -1)
        next_nodes = []
        for b in range(batch_size):
            neighbors = torch.unique(all_neighbors[b])
            scores = torch.gather(
                input=token_logits[b].unsqueeze(0).expand(neighbors.shape[0], -1),
                dim=-1,
                index=(item_id2tokens[neighbors] - 1)
            ).mean(dim=-1)
            idxs = torch.topk(scores, min(num_beams, scores.shape[0])).indices
            next_nodes.append(neighbors[idxs])
        topk_nodes = torch.stack(next_nodes, dim=0)
    return topk_nodes


# ── RPG-ANN (FAISS HNSW graph build) ─────────────────────────────────────────

def build_adjacency_ann(n_items, n_edges, device, n_embd=N_EMBD):
    """
    Simulate build_adjacency_faiss() from RPG_SG/model.py.
    Generates random normalized item embeddings (same shape as real mean-token-emb),
    builds a FAISS HNSW index on CPU, searches n_edges neighbours per item,
    then moves the resulting adjacency to the GPU.
    Memory: O(n * d) CPU + O(n * n_edges) GPU.
    """
    embs = np.random.randn(n_items, n_embd).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.maximum(norms, 1e-8)

    index = faiss.IndexHNSWFlat(n_embd, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(n_edges + 1, 50)
    index.add(embs)

    _, neighbours = index.search(embs, n_edges + 1)
    neighbours = neighbours[:, 1:]            # drop self-match
    neighbours_1idx = neighbours + 1          # 1-indexed item IDs

    adjacency = torch.zeros((n_items + 1, n_edges), dtype=torch.long)
    adjacency[1:] = torch.from_numpy(neighbours_1idx).long()
    adjacency = adjacency.to(device)
    return adjacency


def ann_theoretical_mb(n_items, n_edges, n_embd=N_EMBD):
    """Approximate total memory (MB) consumed by ANN build: CPU FAISS + GPU adjacency."""
    emb_bytes  = n_items * n_embd * 4               # float32 embeddings (CPU)
    hnsw_bytes = n_items * 2 * 32 * 4               # int32 HNSW graph (M=32, bidirectional)
    knn_bytes  = n_items * (n_edges + 1) * 4        # FAISS knn output (int32)
    adj_bytes  = (n_items + 1) * n_edges * 8        # adjacency tensor (int64, GPU)
    return (emb_bytes + hnsw_bytes + knn_bytes + adj_bytes) / 1024 ** 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_item_tokens(n_items, n_digit, codebook_size, device):
    tokens = torch.zeros((n_items + 1, n_digit), dtype=torch.long, device=device)
    for d in range(n_digit):
        offset = d * codebook_size + 1
        tokens[1:, d] = torch.randint(offset, offset + codebook_size,
                                      (n_items,), device=device)
    return tokens


def measure(fn, n_warmup=2, n_repeat=10):
    """Returns (median_ms, peak_gpu_mb) over n_repeat runs."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times, peaks = [], []
    for _ in range(n_repeat):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if torch.cuda.is_available():
            peaks.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

    times.sort()
    peaks.sort()
    return times[len(times) // 2], (peaks[len(peaks) // 2] if peaks else 0.0)


def measure_graph_build(n_items, n_edges, device, n_repeat=3):
    """Time and peak GPU memory for the one-time O(n^2) graph build."""
    times, peaks = [], []
    for _ in range(n_repeat):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        adj = build_graph(n_items, n_edges, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if torch.cuda.is_available():
            peaks.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
        del adj
    times.sort()
    peaks.sort()
    return times[len(times) // 2], (peaks[len(peaks) // 2] if peaks else 0.0)


def measure_ann_build(n_items, n_edges, device, n_repeat=3):
    """Time, peak CPU RSS, and peak GPU memory for the FAISS HNSW build + adjacency upload.
    Uses resource.getrusage for CPU peak — captures the high-water mark during the build."""
    times, gpu_peaks, cpu_peaks = [], [], []
    for _ in range(n_repeat):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
        t0 = time.perf_counter()
        adj = build_adjacency_ann(n_items, n_edges, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if torch.cuda.is_available():
            gpu_peaks.append(torch.cuda.max_memory_allocated() / 1024 ** 2)
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cpu_peaks.append((rss_after - rss_before) / 1024)  # KB -> MB
        del adj
    times.sort()
    gpu_peaks.sort()
    cpu_peaks.sort()
    return (times[len(times) // 2],
            gpu_peaks[len(gpu_peaks) // 2] if gpu_peaks else 0.0,
            cpu_peaks[len(cpu_peaks) // 2])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_digit',           type=int, default=32)
    parser.add_argument('--codebook_size',     type=int, default=256)
    parser.add_argument('--num_beams',         type=int, default=50)
    parser.add_argument('--n_edges',           type=int, default=50)
    parser.add_argument('--propagation_steps', type=int, default=3)
    parser.add_argument('--batch_size',        type=int, default=32)
    parser.add_argument('--n_repeat',          type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Beam search: P={args.num_beams}, E={args.n_edges}, S={args.propagation_steps}')
    print(f'Theoretical per-query crossover: {args.propagation_steps * args.num_beams * args.n_edges:,} items')
    print(f'FAISS available: {HAVE_FAISS}, n_embd: {N_EMBD}\n')

    catalog_sizes = [1_000, 5_000, 10_000, 25_000, 50_000,
                     100_000, 250_000, 500_000, 1_000_000,
                     1_250_000, 1_500_000, 1_750_000, 2_000_000,
                     2_250_000, 2_500_000, 2_750_000, 3_000_000]

    # ── Section 1a: Beam search preprocessing (O(n^2) on GPU) ────────────────
    print('=== SECTION 1: PREPROCESSING (one-time cost) ===')
    print()
    print('  [Beam search] O(n^2) similarity matrix on GPU:')
    print(f'  {"n_items":>10}  {"time (ms)":>12}  {"GPU peak (MB)":>14}  {"n^2 theory (MB)":>16}')
    print('  ' + '-' * 58)

    for n in catalog_sizes:
        theoretical_mb = n * n * 4 / 1024 ** 2
        try:
            t_build, mb_build = measure_graph_build(n, args.n_edges, device, n_repeat=3)
            print(f'  {n:>10,}  {t_build:>12.1f}  {mb_build:>14.1f}  {theoretical_mb:>16.1f}')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  {n:>10,}  {"OOM":>12}  {"OOM":>14}  {theoretical_mb:>16.1f}')

    # ── Section 1b: RPG-ANN preprocessing (FAISS HNSW on CPU) ────────────────
    # Capped at 1M items — HNSW build at 3M with efConstruction=200 takes hours.
    # 1M already demonstrates that ANN never OOMs where beam search hits 250k.
    ann_sizes = [n for n in catalog_sizes if n <= 1_000_000]

    print()
    print('  [RPG-ANN] FAISS HNSW on CPU, adjacency tensor on GPU (capped at 1M):')
    print(f'  {"n_items":>10}  {"time (ms)":>12}  {"CPU peak (MB)":>14}  {"GPU adj (MB)":>13}')
    print('  ' + '-' * 57)

    if not HAVE_FAISS:
        print('  [FAISS not installed — skipping RPG-ANN section]')
    else:
        for n in ann_sizes:
            try:
                t_ann, gpu_mb, cpu_mb = measure_ann_build(n, args.n_edges, device, n_repeat=1)
                print(f'  {n:>10,}  {t_ann:>12.1f}  {cpu_mb:>14.1f}  {gpu_mb:>13.1f}')
            except Exception as e:
                torch.cuda.empty_cache()
                print(f'  {n:>10,}  {"ERR":>12}  {"ERR":>14}  {"ERR":>13}  ({e})')

    # ── Section 2: Per-query inference speed ──────────────────────────────────
    print()
    print('=== SECTION 2: PER-QUERY INFERENCE (after preprocessing) ===')
    print('Note: RPG-ANN and beam search share identical inference (graph propagation).')
    print('      Only the adjacency source differs; per-query latency is the same.')
    print()
    hdr = (f'{"n_items":>10}  {"exh (ms)":>10}  {"exh (MB)":>10}'
           f'  {"beam (ms)":>10}  {"beam (MB)":>10}  {"speedup":>8}')
    print(hdr)
    print('-' * len(hdr))

    for n in catalog_sizes:
        tokens = build_item_tokens(n, args.n_digit, args.codebook_size, device)
        token_logits = torch.randn(
            args.batch_size, args.n_digit * args.codebook_size, device=device
        )

        def run_exhaustive():
            return score_exhaustive(token_logits, tokens, chunk_size=4096)

        try:
            t_exh, mb_exh = measure(run_exhaustive, n_repeat=args.n_repeat)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'{n:>10,}  {"OOM":>10}  {"---":>10}  {"N/A":>10}  {"---":>10}  {"N/A":>8}')
            del tokens
            continue

        # Use random adjacency for beam inference timing (avoids re-running O(n^2) build)
        try:
            adjacency = torch.randint(1, n + 1, (n + 1, args.n_edges),
                                      dtype=torch.long, device=device)

            def run_beam():
                return score_beam(token_logits, tokens, adjacency,
                                  args.num_beams, args.propagation_steps, device)

            t_beam, mb_beam = measure(run_beam, n_repeat=args.n_repeat)
            del adjacency
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_beam, mb_beam = float('nan'), float('nan')

        speedup = t_exh / t_beam if t_beam == t_beam else float('nan')
        winner = '<-- beam faster' if speedup > 1 else '<-- exhaustive faster'
        print(f'{n:>10,}  {t_exh:>10.2f}  {mb_exh:>10.1f}'
              f'  {t_beam:>10.2f}  {mb_beam:>10.1f}  {speedup:>8.2f}x  {winner}')
        del tokens


if __name__ == '__main__':
    main()
