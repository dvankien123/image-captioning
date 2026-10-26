import torch
import torch.nn.functional as F


def length_penalty(length, alpha):
    """Return the GNMT length penalty for emitted tokens (excluding <start>)."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    decoder,
    features,
    vocab,
    beam_size=3,
    max_len=30,
    length_penalty_alpha=0.6,
    return_alphas=False,
):
    """Decode one image feature map using beam search with Transformer Decoder.

    Args:
        decoder: TransformerDecoder instance.
        features: Visual features of shape (1, 197, d_model).
        vocab: Vocabulary object.
        beam_size (int): Number of beams to track.
        max_len (int): Maximum length of generated caption.
        length_penalty_alpha (float): GNMT length penalty factor.
        return_alphas (bool): Whether to return cross-attention heatmaps.

    Returns:
        best_ids (list[int]): Generated word IDs (without <start> and <end>).
        alpha_tensor (torch.Tensor, optional): Stacked cross-attention weights.
    """
    if features.size(0) != 1:
        raise ValueError("beam_search expects features for exactly one image")
    if beam_size < 1 or max_len < 1:
        raise ValueError("beam_size and max_len must be positive")
    if length_penalty_alpha < 0:
        raise ValueError("length_penalty_alpha must be non-negative")

    device = features.device
    start_id = vocab.word2idx["<start>"]
    end_id = vocab.word2idx["<end>"]
    pad_id = vocab.word2idx["<pad>"]

    # Each beam: (token_ids, cumulative_log_prob, length, alphas_list)
    # token_ids starts with [<start>]
    beams = [([start_id], 0.0, 0, [])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, score, length, alphas in beams:
            if token_ids and token_ids[-1] == end_id:
                completed.append((token_ids, score, length, alphas))
                continue

            current_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
            logits, step_alpha = decoder.forward_step(features, current_tokens)

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            log_probs[start_id] = float("-inf")
            log_probs[pad_id] = float("-inf")

            values, indices = log_probs.topk(min(beam_size, log_probs.numel()))

            for value, word_id in zip(values.tolist(), indices.tolist()):
                step_alpha_cpu = (
                    step_alpha.squeeze(0).detach().cpu()
                    if step_alpha is not None
                    else torch.zeros(features.size(1))
                )
                candidates.append((
                    token_ids + [word_id],
                    score + value,
                    length + 1,
                    alphas + [step_alpha_cpu],
                ))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

    final_candidates = completed + beams
    if not final_candidates:
        return ([], torch.empty(0, features.size(1))) if return_alphas else []

    best_tokens, _, _, best_alphas = max(
        final_candidates,
        key=lambda item: item[1] / length_penalty(item[2], length_penalty_alpha),
    )

    # Strip <start>
    if best_tokens and best_tokens[0] == start_id:
        best_tokens = best_tokens[1:]

    # Strip <end> and truncate alphas accordingly
    if end_id in best_tokens:
        idx = best_tokens.index(end_id)
        best_tokens = best_tokens[:idx]
        best_alphas = best_alphas[:idx]

    if return_alphas:
        alpha_tensor = torch.stack(best_alphas) if best_alphas else torch.empty(0, features.size(1))
        return best_tokens, alpha_tensor
    return best_tokens
