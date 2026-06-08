import torch
from typing import Dict, List, Tuple, Callable
from transformer_lens import HookedTransformer
from functools import partial

def get_node_metadata(component_name: str) -> Tuple[str, int, slice]:
    """
    Parses a component name (e.g., 'a10.h3' or 'm5') into HookedTransformer hook details.
    
    Returns:
        hook_name: The TL hook name.
        layer_idx: The layer number.
        slice_idx: The slice index (head index for attn, None for MLP).
    """
    if component_name.startswith('m'):
        # MLP Node: m{layer}
        layer = int(component_name[1:])
        return f"blocks.{layer}.hook_mlp_out", layer, slice(None)
    elif component_name.startswith('a'):
        # Attention Head: a{layer}.h{head}
        parts = component_name.split('.')
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        return f"blocks.{layer}.attn.hook_result", layer, head
    elif component_name == 'logits':
        return "logits", -1, slice(None)
    else:
        raise ValueError(f"Unknown component format: {component_name}")

def get_faithfulness_hooks(
    model: HookedTransformer, 
    mask: torch.Tensor, 
    large_component_to_idx: Dict[str, int],
    cache_clean: Dict[str, torch.Tensor]
) -> List[Tuple[str, Callable]]:
    """
    Constructs hooks for differentiable patching based on a soft mask.
    
    The intervention is:
    Activation_new = mask * Activation_clean + (1 - mask) * Activation_corrupt
    
    Since we run the model on the CORRUPT input, Activation_corrupt is the default 
    tensor passed to the hook. We basically inject the clean activation weighted by 'mask'.
    
    Mathematically equivalent to:
    Activation_new = Activation_corrupt + mask * (Activation_clean - Activation_corrupt)
    
    Args:
        model: The Large HookedTransformer.
        mask: Tensor of shape (n_large,) containing soft mask values [0,1].
              Must be on the same device as the model.
        large_component_to_idx: Mapping from component name to index in 'mask'.
        cache_clean: Cache dictionary containing activations from the clean run.
    
    Returns:
        List of (hook_name, hook_fn) tuples.
    """
    hooks = []
    
    # Group components by hook_name to avoid adding multiple hooks to the same point
    hook_points = {} # hook_name -> list of (component_name, slice_idx, mask_idx)
    
    for name, idx in large_component_to_idx.items():
        if name in ['input', 'logits']: continue # Skip input/logits for patching usually
        
        hook_name, layer, slice_idx = get_node_metadata(name)
        
        if hook_name not in hook_points:
            hook_points[hook_name] = []
        hook_points[hook_name].append((name, slice_idx, idx))
        
    def generic_patching_hook(activations: torch.Tensor, hook, points_info):
        """
        points_info: List of (name, slice_idx, mask_idx)
        """
        # activations here are from the CORRUPT run
        clean_act = cache_clean[hook.name]
        
        # We need to compute delta = mask * (clean - corrupt)
        # But we must do it carefully to support efficient broadcasting
        
        for _, slice_idx, mask_idx in points_info:
            m = mask[mask_idx] # Scalar (or 0-d tensor) with grad
            
            # Extract relevant parts
            if isinstance(slice_idx, int):
                # Attention Head: [batch, pos, head, d_head]
                # We patch specific head 'slice_idx'
                
                # Note: m is a scalar for this component. 
                # We construct the mixed activation.
                # Act_new = m * Clean + (1-m) * Corrupt
                # Act_new = Corrupt + m * (Clean - Corrupt)
                
                diff = clean_act[:, :, slice_idx, :] - activations[:, :, slice_idx, :]
                activations[:, :, slice_idx, :] = activations[:, :, slice_idx, :] + m * diff
                
            else:
                # MLP: [batch, pos, d_model]
                # We patch the whole MLP
                diff = clean_act - activations
                activations[:] = activations + m * diff
                
        return activations

    # Create the actual hook functions using partial to bind the specific points info
    for hook_name, points_info in hook_points.items():
        hook_fn = partial(generic_patching_hook, points_info=points_info)
        hooks.append((hook_name, hook_fn))
        
    return hooks