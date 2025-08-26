import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor): The logits distribution of shape (vocab_size,).
        top_k (int): Keep only top k tokens with highest probability (0 = no filtering).
        top_p (float): Keep the top tokens with cumulative probability >= top_p.

    Returns:
        torch.Tensor: Filtered logits.
    """
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.tensor(float('-inf')).to(logits.device), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')

    return logits


def decode(
    input_ids: torch.Tensor,
    position: int,
    model: torch.nn.Module,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    apply_softmax: bool = False,
    top_k: int = 0,
    top_p: float = 1.0,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Decodes the next token's logits (or probabilities) from the model.

    Args:
        input_ids (torch.Tensor): Tokenized input sequence of shape (1, seq_len).
        position (int): The current position (token index) in generation.
        model (torch.nn.Module): Transformer model used for decoding.
        past_key_values (Tuple, optional): Cached keys/values for speedup (default: None).
        apply_softmax (bool): Whether to return softmax probabilities instead of raw logits.
        top_k (int): Top-K filtering for logits (0 = disable).
        top_p (float): Top-P (nucleus) filtering (1.0 = disable).
        device (str | torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Logits or probabilities for next-token prediction.
    """
    input_ids = input_ids.to(device)
    if past_key_values:
        past_key_values = tuple(pk.to(device) for pk in past_key_values)

    logger.info(f"🧠 [decode] Running inference at position: {position}")
    logger.debug(f"📥 input_ids shape: {input_ids.shape}")
    logger.debug(f"🔁 past_key_values: {'Provided' if past_key_values else 'None'}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)

    logger.debug(f"📤 Raw logits shape: {logits.shape}")

    # Apply filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    if apply_softmax:
        probs = F.softmax(logits, dim=-1)
        logger.info(f"✅ Returned softmax probabilities.")
        return probs

    logger.info(f"✅ Returned raw logits.")
    return logits
print("kernel.py loaded")
print("act_quant defined:", "act_quant" in dir())
