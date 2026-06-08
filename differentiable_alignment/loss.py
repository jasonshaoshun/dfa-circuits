import torch
import torch.nn.functional as F

def faithfulness_loss(
    clean_logits: torch.Tensor, 
    intervened_logits: torch.Tensor, 
    mask: torch.Tensor,
    lambda_sparsity: float = 0.01,
    sparsity_type: str = 'l1'
) -> torch.Tensor:
    """
    Computes the Differentiable Faithfulness Loss.
    
    L = KL(P_clean || P_intervened) + lambda * Sparsity(mask)
    
    Args:
        clean_logits: Logits from the clean forward pass.
        intervened_logits: Logits from the intervened (soft-patched) pass.
        mask: The soft mask vector predicted by the alignment model.
        lambda_sparsity: Coefficient for sparsity penalty.
        sparsity_type: 'l1' or 'entropy'.
    
    Returns:
        scalar loss tensor.
    """
    # 1. KL Divergence
    # KL(P || Q) = sum(P * (log P - log Q))
    # PyTorch kl_div expects input to be log-probabilities (Q) and target to be probs (P)
    clean_logits = clean_logits.float()
    intervened_logits = intervened_logits.float()
    
    log_probs_intervened = F.log_softmax(intervened_logits, dim=-1)
    probs_clean = F.softmax(clean_logits, dim=-1)
    
    # batchmean gives the correct mathematical KL averaged over the batch
    kl_loss = F.kl_div(log_probs_intervened, probs_clean, reduction='batchmean')
    
    # 2. Sparsity Loss
    if sparsity_type == 'l1':
        sparsity_loss = torch.mean(torch.abs(mask))
    elif sparsity_type == 'entropy':
        # Entropy of the mask values treated as probabilities? 
        # Usually for mask sparsity we want values pushed to 0 or 1, 
        # or just sum(mask) minimized (L1).
        # If we want binary-like masks, we might use entropy = -sum(m log m + (1-m) log(1-m))
        # But standard L1 is safest for general sparsity.
        sparsity_loss = torch.mean(torch.abs(mask))
    else:
        sparsity_loss = torch.tensor(0.0, device=mask.device)
        
    total_loss = kl_loss + lambda_sparsity * sparsity_loss
    
    return total_loss, kl_loss, sparsity_loss
