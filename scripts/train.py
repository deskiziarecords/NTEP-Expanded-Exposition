# scripts/train.py
# Minimal training script for NTEP spherical embeddings.
# Implements the loss described in axiom **A2** (order preservation).

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def cosine_sim(u, v):
    """Cosine similarity between two tensors."""
    return F.cosine_similarity(u, v)

def contrastive_loss(embeddings, pos_pairs, neg_pairs, gamma=0.82, margin_pos=0.0, margin_neg=-0.5):
    """
    embeddings: Tensor of shape (N, D) – already L2‑normalized.
    pos_pairs: List of (i, j) indices that must satisfy cos ≥ γ.
    neg_pairs: List of (i, j) indices that must satisfy cos ≤ margin_neg.
    """
    loss = 0.0
    # Positive pairs (must be > γ)
    for i, j in pos_pairs:
        sim = cosine_sim(embeddings[i], embeddings[j])
        loss += F.relu(margin_pos - sim)  # push similarity upward

    # Negative pairs (must be < margin_neg)
    for i, j in neg_pairs:
        sim = cosine_sim(embeddings[i], embeddings[j])
        loss += F.relu(sim - margin_neg)  # push similarity downward
    return loss

# -------------------------------------------------
# Dataset stub (replace with real metadata)
# -------------------------------------------------
class ToolDataset(Dataset):
    def __init__(self, tool_ids, pos_pairs, neg_pairs):
        self.tool_ids = tool_ids
        self.pos_pairs = pos_pairs
        self.neg_pairs = neg_pairs

    def __len__(self):
        return len(self.tool_ids)

    def __getitem__(self, idx):
        # In a real setting you would load pre‑computed embeddings.
        # Here we just return the index; the training loop will fetch the vector.
        return self.tool_ids[idx]

# -------------------------------------------------
# Main training loop
# -------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = args.embed_dim  # 768 per spec

    # Mock tool IDs (0 … N‑1)
    N = args.num_tools
    tool_ids = torch.arange(N, dtype=torch.long)

    # Dummy positive / negative pair lists (populate with real semantics)
    # For demo purposes we assume a simple chain t0 ≤ t1 ≤ t2 …
    pos_pairs = [(i, i+1) for i in range(N-1)]
    # Non‑adjacent pairs as negatives
    neg_pairs = [(i, j) for i in range(N) for j in range(i+2, N)]

    dataset = ToolDataset(tool_ids, pos_pairs, neg_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Simple linear projection + normalization = spherical embedding
    encoder = nn.Sequential(
        nn.Linear(args.input_dim, D),
        nn.Lambda(lambda x: F.normalize(x, p=2, dim=-1))  # unit‑norm
    )

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            # batch is just a list of indices; fetch their *random* raw vectors
            # (in practice you would feed real feature vectors)
            raw = torch.randn(batch.shape[0], args.input_dim, device=device)
            emb = encoder(raw)                     # (B, D), unit‑norm
            # Here we repeat embeddings across the batch to compute pair losses
            # For brevity, we compute loss on the whole dataset each step:
        # Compute full‑batch loss
        all_emb = encoder(torch.randn(N, args.input_dim, device=device))
        loss = contrastive_loss(
            embeddings=all_emb,
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            gamma=args.gamma,
            margin_pos=0.0,
            margin_neg=args.margin_neg
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{args.epochs} – Loss: {epoch_loss/len(loader):.4f}')

    # Save normalized embeddings to data/embeddings.pt
    torch.save(encoder(torch.randn(N, args.input_dim)), args.output_path)
    print(f'Saved embeddings to {args.output_path}')

# -------------------------------------------------
# Argument parsing
# -------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NTEP spherical embeddings')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=768, help='Target dimensionality (default: 768)')
    parser.add_argument('--input_dim', type=int, default=1024, help='Dimensionality of raw input features')
    parser.add_argument('--num_tools', type=int, default=5, help='Number of atomic tools')
    parser.add_argument('--gamma', type=float, default=0.82, help='Comparability constant γ')
    parser.add_argument('--margin_neg', type=float, default=-0.5, help='Margin for negative pairs')
    parser.add_argument('--output_path', type=str, default='data/embeddings.pt', help='File to save embeddings')
    args = parser.parse_args()
    main(args)
