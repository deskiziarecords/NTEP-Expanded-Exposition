import numpy as np
import torch

def composite_vector(embeddings: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """Calculates the normalized sum (composite vector) of embeddings."""
    if weights is None:
        # Uniform weighting
        weights = torch.ones(embeddings.shape[0]) / embeddings.shape[0]
    
    weighted_sum = torch.sum(embeddings * weights.unsqueeze(-1), dim=0)
    return torch.nn.functional.normalize(weighted_sum, p=2, dim=0)

def calculate_delta(gamma: float, k: int) -> float:
    """Quantifies the minimum separation for a chain of length k."""
    return 2 * (1 - gamma) / k

def check_monotonicity(chain_embeddings: torch.Tensor, gamma: float) -> bool:
    """
    Checks if the Monotonicity Theorem holds for a given chain of embeddings.
    Returns True if the chain is geometrically monotonic.
    """
    k = chain_embeddings.shape[0]
    delta = calculate_delta(gamma, k)
    
    v = composite_vector(chain_embeddings)
    
    # Calculate cosine similarities
    similarities = torch.mv(chain_embeddings, v)
    
    # Check if for all i < j, sim(v, t_i) >= sim(v, t_j) + delta
    for i in range(k):
        for j in range(i + 1, k):
            if similarities[i] < similarities[j] + delta - 1e-9: # small epsilon for float error
                # print(f"Monotonicity failed between {i} and {j}")
                # print(f"Sim(i)={similarities[i]}, Sim(j)={similarities[j]}, Delta={delta}")
                return False
    return True
