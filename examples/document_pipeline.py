import torch
import numpy as np

from src.ntep import Tool, DependencyGraph
from src.ntep.core.embedding import ToolEncoder
from src.ntep.math.geometry import check_monotonicity, calculate_delta

# --- 1. Mock Setup (In a real repo, this would be loaded from files) ---

def create_mock_setup():
    tools = [
        Tool(id="resize", name="Resize Image", description="Resizes an image to a specified dimension."),
        Tool(id="encrypt", name="Encrypt File", description="Applies AES-256 encryption to a file."),
        Tool(id="package", name="Package for Delivery", description="Packages files into a zip archive."),
        Tool(id="email", name="Send via Email", description="Sends the packaged file via email.")
    ]
    
    dependencies = {
        "resize": ["package"],
        "encrypt": ["package"],
        "package": ["email"]
    }
    dep_graph = DependencyGraph(dependencies)
    
    # Mock an encoder by creating pre-baked embeddings that satisfy the constraints
    # In reality, this would be a SentenceTransformer model.
    class MockEncoder:
        def __init__(self):
            # Create embeddings that are close for dependent pairs and far for others
            np.random.seed(42)
            dim = 768
            self.embeds = {
                "resize": self._normalize(np.random.randn(dim)),
                "encrypt": self._normalize(np.random.randn(dim)),
                # Make 'package' close to 'resize' and 'encrypt'
                "package": self._normalize(0.8 * self.embeds["resize"] + 0.2 * self._normalize(np.random.randn(dim))),
                # Make 'email' close to 'package'
                "email": self._normalize(0.9 * self.embeds["package"] + 0.1 * self._normalize(np.random.randn(dim))),
            }
        
        def _normalize(self, v):
            return v / np.linalg.norm(v)

        def encode(self, texts):
            # Simple mock: return embedding based on keywords
            result = []
            for text in text:
                if "resize" in text.lower(): result.append(torch.tensor(self.embeds["resize"]))
                elif "encrypt" in text.lower(): result.append(torch.tensor(self.embeds["encrypt"]))
                elif "package" in text.lower(): result.append(torch.tensor(self.embeds["package"]))
                elif "email" in text.lower(): result.append(torch.tensor(self.embeds["email"]))
                else: # Default to a random vector for prompts
                    result.append(torch.tensor(self._normalize(np.random.randn(768))))
            return torch.stack(result)

    encoder = MockEncoder()
    return tools, dep_graph, encoder

# --- 2. Run the Example ---

if __name__ == "__main__":
    tools, dep_graph, encoder = create_mock_setup()
    
    # --- Test the Monotonicity Theorem ---
    print("--- Testing Monotonicity Theorem ---")
    valid_chain_ids = ["resize", "package", "email"]
    chain_embeddings = encoder.encode(valid_chain_ids)
    
    gamma = 0.7 # A reasonable guess for our mock embeddings
    is_monotonic = check_monotonicity(chain_embeddings, gamma)
    delta = calculate_delta(gamma, len(valid_chain_ids))
    
    print(f"Chain: {' -> '.join(valid_chain_ids)}")
    print(f"Gamma (γ): {gamma:.2f}")
    print(f"Calculated Delta (δ): {delta:.4f}")
    print(f"Is the chain geometrically monotonic? {is_monotonic}\n")

    # --- Test the Decoder ---
    print("--- Testing NTEP Decoder ---")
    from src.ntep.inference.decoder import NTEPDecoder
    
    decoder = NTEPDecoder(tools, dep_graph, encoder, gamma=gamma)
    prompt_text = "Prepare the confidential image for delivery"
    prompt_embedding = decoder.encode_prompt(prompt_text)
    
    decoded_chain = decoder.decode(prompt_embedding)
    
    print(f"Prompt: '{prompt_text}'")
    print(f"Decoded Chain: {' -> '.join(decoded_chain)}")
    print(f"Is the decoded chain valid? {dep_graph.is_valid_chain(decoded_chain)}")
