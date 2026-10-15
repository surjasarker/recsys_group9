# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """Residual block used as a per-codebook prediction head. Initialized as identity."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)  # start as identity mapping
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        self.pred_heads = nn.Sequential(*[ResBlock(self.config['n_embd']) for _ in range(self.n_pred_head)])

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        # Cluster-based decoding state
        self.generate_w_decoding_graph = False  # flag set by pipeline before test eval
        self.init_flag = False
        self.n_clusters = config['n_clusters']
        self.n_probe_clusters = config['n_probe_clusters']

    def _map_item_tokens(self) -> torch.Tensor:
        """Build a (n_items, n_digit) lookup table mapping item IDs to their semantic token IDs."""
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (
            f'#Embedding parameters: {emb_params}\n'
            f'#Non-embedding parameters: {total_params - emb_params}\n'
            f'#Total trainable parameters: {total_params}\n'
        )

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """GPT-2 forward pass. Returns model outputs with loss attached when return_loss=True."""
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [
                torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    # ------------------------------------------------------------------
    # Cluster-based decoding — precomputation
    # ------------------------------------------------------------------

    def _build_item_profiles(self) -> np.ndarray:
        """
        Binary indicator matrix: profile[i, j] = 1 if item (i+1) uses flat token j.
        Shape: (n_items-1, n_digit * codebook_size), float32.
        Used only transiently during init_clusters; not stored.
        """
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        total_tokens = n_digit * codebook_size
        n_real_items = self.dataset.n_items - 1  # exclude padding item 0

        # item_id2tokens[1:]: (n_real_items, n_digit), 1-indexed flat token ids
        # subtract 1 → 0-indexed positions in [0, total_tokens)
        flat_ids = self.item_id2tokens[1:].cpu().numpy() - 1  # (n_real_items, n_digit)

        profiles = np.zeros((n_real_items, total_tokens), dtype=np.float32)
        row_idx = np.arange(n_real_items)[:, None]  # broadcasts over n_digit
        profiles[row_idx, flat_ids] = 1.0
        return profiles

    def init_clusters(self):
        """
        Precompute everything needed for cluster-based decoding:
          - cluster_centroids:  (n_clusters, total_tokens) on GPU
          - cluster_item_ids:   (n_clusters, max_cluster_size) padded, on GPU
          - cluster_item_mask:  (n_clusters, max_cluster_size) bool, on GPU
        Stores self.cluster_init_time_s for downstream latency reporting.
        """
        import faiss
        import time
        _t0 = time.perf_counter()

        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        total_tokens = n_digit * codebook_size
        n_real_items = self.dataset.n_items - 1
        device = self.config['device']

        self.tokenizer.log("Building item profiles...")
        profiles = self._build_item_profiles()  # (n_real_items, total_tokens)

        # ------ k-means clustering ------
        self.tokenizer.log(f"Running k-means with {self.n_clusters} clusters...")
        kmeans = faiss.Kmeans(
            d=total_tokens,
            k=self.n_clusters,
            niter=20,
            seed=self.config['rand_seed'],
            gpu=self.config.get('opq_use_gpu', False),
            verbose=False,
        )
        kmeans.train(profiles)

        # centroids: (n_clusters, total_tokens) — each entry is fraction of cluster items owning that token
        self.cluster_centroids = torch.from_numpy(kmeans.centroids).to(dtype=torch.float32, device=device)

        # ------ assign items to clusters ------
        _, assignments = kmeans.index.search(profiles, 1)
        assignments = assignments.squeeze(1).tolist()  # length n_real_items

        # build padded cluster → item_id (1-indexed) lookup
        cluster_lists = [[] for _ in range(self.n_clusters)]
        for local_idx, cluster_id in enumerate(assignments):
            cluster_lists[cluster_id].append(local_idx + 1)  # 1-indexed item_id

        max_size = max((len(c) for c in cluster_lists), default=1)
        cluster_item_ids = torch.zeros((self.n_clusters, max_size), dtype=torch.long)
        cluster_item_mask = torch.zeros((self.n_clusters, max_size), dtype=torch.bool)
        for c, items in enumerate(cluster_lists):
            if items:
                cluster_item_ids[c, :len(items)] = torch.tensor(items, dtype=torch.long)
                cluster_item_mask[c, :len(items)] = True

        self.cluster_item_ids = cluster_item_ids.to(device)
        self.cluster_item_mask = cluster_item_mask.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.cluster_init_time_s = time.perf_counter() - _t0

        avg_size = n_real_items / self.n_clusters
        self.tokenizer.log(
            f"Cluster search ready: {self.n_clusters} clusters, "
            f"avg {avg_size:.1f} items/cluster, max {max_size} items/cluster, "
            f"init took {self.cluster_init_time_s:.2f}s"
        )

    # ------------------------------------------------------------------
    # Cluster-based decoding — inference
    # ------------------------------------------------------------------

    def cluster_search(self, token_logits: torch.Tensor, n_return_sequences: int):
        """
        1. Score all clusters: one matmul.
        2. Gather candidate items from top-n_probe_clusters clusters.
        3. Score candidates: index into token_logits by precomputed token positions, mean over digits.
        4. Return top-n_return_sequences items.

        Args:
            token_logits: (batch, n_digit * codebook_size) — concatenated log-softmax log-probs
        Returns:
            preds:     (batch, n_return_sequences, 1) — 1-indexed item IDs
            n_visited: (batch, 1)                     — number of candidates scored
        """
        batch_size = token_logits.shape[0]
        n_probe = min(self.n_probe_clusters, self.n_clusters)
        max_size = self.cluster_item_ids.shape[1]

        # ---- Step 1: score clusters ----
        cluster_scores = torch.matmul(token_logits, self.cluster_centroids.T)  # (batch, n_clusters)
        top_cluster_ids = cluster_scores.topk(n_probe, dim=-1).indices          # (batch, n_probe)

        # ---- Step 2: gather candidate item IDs and mask ----
        cand_ids  = self.cluster_item_ids[top_cluster_ids]    # (batch, n_probe, max_size)
        cand_mask = self.cluster_item_mask[top_cluster_ids]   # (batch, n_probe, max_size)

        n_cands = n_probe * max_size
        cand_ids  = cand_ids.reshape(batch_size, n_cands)   # (batch, n_cands)
        cand_mask = cand_mask.reshape(batch_size, n_cands)  # (batch, n_cands)

        # ---- Step 3: score candidates via token_logits lookup ----
        # flat_token_ids[b, i, d] = 0-indexed position in token_logits for item cand_ids[b,i], digit d
        flat_token_ids = (self.item_id2tokens[cand_ids] - 1).clamp(min=0)  # (batch, n_cands, n_digit)

        # gather log-probs and average across digits → item score
        b_idx = torch.arange(batch_size, device=token_logits.device).view(-1, 1, 1).expand_as(flat_token_ids)
        item_scores = token_logits[b_idx, flat_token_ids].mean(dim=-1)  # (batch, n_cands)

        # mask padded slots
        item_scores = item_scores.masked_fill(~cand_mask, float('-inf'))

        # ---- Step 4: top-k ----
        topk_local = item_scores.topk(n_return_sequences, dim=-1).indices  # (batch, k)
        preds = cand_ids.gather(dim=-1, index=topk_local)                  # (batch, k)

        n_visited = cand_mask.sum(dim=-1, keepdim=True).float()  # (batch, 1)
        return preds.unsqueeze(-1), n_visited

    def generate(self, batch, n_return_sequences=1):
        """
        At test time: routes through cluster_search() if generate_w_decoding_graph=True,
        otherwise falls back to exhaustive scoring. Clusters are built lazily on first call.
        """
        outputs = self.forward(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:, 0, i, :], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)  # (batch, n_digit * codebook_size)

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_clusters()
                self.init_flag = True
            return self.cluster_search(token_logits, n_return_sequences)

        # Brute-force: score all items (used at validation time)
        # Gather log-probs for each item's tokens then average over digits.
        # flat_ids: (n_real_items, n_digit) — 0-indexed positions in token_logits
        flat_ids = (self.item_id2tokens[1:] - 1)  # (n_real_items, n_digit)
        # token_logits[:, flat_ids]: (batch, n_real_items, n_digit)
        item_scores = token_logits[:, flat_ids].mean(dim=-1)  # (batch, n_real_items)
        preds = item_scores.topk(n_return_sequences, dim=-1).indices + 1  # 1-indexed
        return preds.unsqueeze(-1)
