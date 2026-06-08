import torch
import torch.nn as nn
from typing import List, Dict

class AlignmentModel(nn.Module):
    def __init__(self, 
                 small_component_names: List[str], 
                 large_component_names: List[str],
                 init_scale: float = 0.02):
        """
        Differentiable Alignment Model learning a matrix W between small and large model components.
        
        Args:
            small_component_names: List of node names for the small model (e.g., ['a0.h0', 'm0', ...])
            large_component_names: List of node names for the large model.
            init_scale: Scaling factor for random initialization.
        """
        super().__init__()
        self.small_component_names = small_component_names
        self.large_component_names = large_component_names
        
        self.n_small = len(small_component_names)
        self.n_large = len(large_component_names)
        
        # We assume W lives on the same device as the large model eventually
        self.W = nn.Parameter(torch.randn(self.n_small, self.n_large) * init_scale)

    def predict_mask(self, small_scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Project small model scores to a soft mask for the large model.
        
        Args:
            small_scores: Tensor of shape (n_small,) or (batch, n_small).
                          These are the importance scores 's'.
            temperature: Temperature for the sigmoid function (optional).
            
        Returns:
            large_mask: Tensor of shape (n_large,) or (batch, n_large) with values in [0, 1].
        """
        # Linear projection: s @ W
        # shape: (..., n_small) @ (n_small, n_large) -> (..., n_large)
        logits = torch.matmul(small_scores, self.W)
        
        # Sigmoid activation to get soft mask m
        mask = torch.sigmoid(logits / temperature)
        
        return mask

    def get_large_component_mapping(self) -> Dict[str, int]:
        """Returns a mapping from large component name to index in the mask."""
        return {name: i for i, name in enumerate(self.large_component_names)}