import torch
import numpy as np
from typing import List, Set

from ..core.tool import Tool
from ..core.dependency import DependencyGraph
from ..core.embedding import ToolEncoder # Assuming you have this
from ..math.geometry import composite_vector

class NTEPDecoder:
    def __init__(self, tools: List[Tool], dep_graph: DependencyGraph, encoder: ToolEncoder, gamma: float = 0.8):
        self.tools = {t.id: t for t in tools}
        self.dep_graph = dep_graph
        self.encoder = encoder
        self.gamma = gamma
        
        # Pre-encode all tools
        tool_descriptions = [t.description for t in self.tools.values()]
        self.embeddings = self.encoder.encode(tool_descriptions)
        self.tool_ids = list(self.tools.keys())

    @classmethod
    def from_pretrained(cls, model_path: str):
        # In a real scenario, this would load tools, deps, and encoder weights from a file
        # For this example, we'll mock it.
        print(f"Loading mock model from {model_path}...")
        from ...examples.document_pipeline import create_mock_setup
        tools, dep_graph, encoder = create_mock_setup()
        return cls(tools, dep_graph, encoder)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encodes a text prompt into the embedding space."""
        return self.encoder.encode([prompt])[0]

    def decode(self, prompt_embedding: torch.Tensor) -> List[str]:
        """
        Decodes a valid tool chain from a prompt embedding.
        """
        remaining_tools = set(self.tool_ids)
        executed_chain = []
        
        while remaining_tools:
            # 1. Find all valid candidates
            candidates = [
                t_id for t_id in remaining_tools 
                if self.dep_graph.is_valid_next_step(t_id, set(executed_chain))
            ]
            
            if not candidates:
                raise ValueError("No valid next tool found. Dependency graph might be cyclic or incomplete.")

            # 2. Get embeddings for candidates
            candidate_indices = [self.tool_ids.index(t_id) for t_id in candidates]
            candidate_embeddings = self.embeddings[candidate_indices]
            
            # 3. Find the best candidate
            # Combine prompt with current state (optional but powerful)
            if not executed_chain:
                current_state_embedding = prompt_embedding
            else:
                executed_indices = [self.tool_ids.index(t_id) for t_id in executed_chain]
                state_composite = composite_vector(self.embeddings[executed_indices])
                # Simple averaging of prompt and state
                current_state_embedding = torch.nn.functional.normalize(prompt_embedding + state_composite, p=2, dim=0)

            similarities = torch.mv(candidate_embeddings, current_state_embedding)
            best_idx = torch.argmax(similarities)
            next_tool_id = candidates[best_idx]
            
            # 4. Update and loop
            executed_chain.append(next_tool_id)
            remaining_tools.remove(next_tool_id)
            
        return executed_chain
