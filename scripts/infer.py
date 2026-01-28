# scripts/infer.py
# Decodes a valid tool chain from an external prompt embedding
# using the `decode_chain` routine described in the Zenodo text.

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# -------------------------------------------------
# Load pre‑trained embeddings
# -------------------------------------------------
def load_embeddings(path: Path):
    """Return a dict {tool_name: torch.Tensor} of unit‑norm vectors."""
    checkpoint = torch.load(path, map_location='cpu')
    # For the demo we assume `checkpoint` is already a mapping.
    # In practice the file stores a tensor of shape (N, 768).
    return checkpoint

# -------------------------------------------------
# Simple cosine similarity utility
# -------------------------------------------------
def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1)

# -------------------------------------------------
# Composite vector computation (normalized average)
# -------------------------------------------------
def composite_vector(vectors):
    """vectors: Tensor of shape (k, D). Returns unit‑norm sum / k."""
    avg = torch.mean(vectors, dim=0)
    return F.normalize(avg, p=2)

# -------------------------------------------------
# Chain decoder (core of soundness‑by‑construction)
# -------------------------------------------------
def decode_chain(tools, deps, prompt_emb, embeddings):
    """
    tools: list of tool names (order irrelevant)
    deps: dict mapping tool -> list of prerequisites
    prompt_emb: Tensor of shape (D,) – external prompt representation
    embeddings: dict {tool: Tensor(D)} – spherical embeddings
    Returns: list of tools in execution order.
    """
    remaining = set(tools)
    chain = []

    while remaining:
        # 1️⃣ Composite of already placed tools (empty at first iteration)
        placed_vecs = torch.stack([embeddings[t] for t in chain]) if chain else torch.zeros(1, embeddings[list(tools)[0]].shape[0])
        comp = composite_vector(placed_vecs)          # unit‑norm

        # 2️⃣ Combine with prompt – we simply add (could be weighted)
        v = comp + prompt_emb
        v = F.normalize(v, p=2)

        # 3️⃣ Candidate set must respect all prerequisites
        candidates = [
            t for t in remaining
            if all(pre in chain for pre in deps[t])
        ]

        # 4️⃣ Pick tool with maximal cosine similarity to v
        best_tool = max(candidates, key=lambda t: cosine(v, embeddings[t]).item())

        chain.append(best_tool)
        remaining.remove(best_tool)

    return chain

# -------------------------------------------------
# CLI entry point
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Decode a tool chain')
    parser.add_argument('--prompt', type=str, required=True, help='Short textual prompt')
    parser.add_argument('--chain', type=str, required=True, help='Space‑separated list of tools, e.g. "resize package email"')
    parser.add_argument('--embeddings', type=Path, default='data/embeddings.pt', help='Path to saved embeddings')
    args = parser.parse_args()

    # Simulate a prompt embedding (normally you would obtain this from a language model)
    torch.manual_seed(42)
    prompt_vec = torch.randn(768)
    prompt_vec = F.normalize(prompt_vec, p=2)

    # Load embeddings
    checkpoint = load_embeddings(args.embeddings)
    # Convert to a dict {tool_name: tensor}
    # Here we assume the checkpoint file already contains a dict keyed by integer indices.
    # For the demonstration we just create dummy names:
    dummy_names = ['resize', 'encrypt', 'package', 'email']
    embeddings = {name: checkpoint[i] for i, name in enumerate(dummy_names)}

    # Build dependency map (thin poset)
    deps = {
        'resize': [],
        'encrypt': [],
        'package': ['resize', 'encrypt'],
        'email': ['package']
    }

    # Parse the requested chain (only for printing; decoder recomputes it)
    user_chain = args.chain.split()
    print('User‑provided chain order:', user_chain)

    # Decode using our algorithm
    decoded = decode_chain(tools=user_chain, deps=deps, prompt_emb=prompt_vec, embeddings=embeddings)
    print('Decoded execution order (soundness‑guaranteed):', decoded)

if __name__ == '__main__':
    main()
