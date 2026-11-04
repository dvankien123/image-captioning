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
    return_nbest=False,
    nbest=4,
    token_top_k=6,
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
        return_nbest (bool): Return the best complete hypotheses with token details.
        nbest (int): Maximum number of hypotheses returned when ``return_nbest``.
        token_top_k (int): Number of next-token alternatives retained per position.

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
    if nbest < 1 or token_top_k < 1:
        raise ValueError("nbest and token_top_k must be positive")

    device = features.device
    start_id = vocab.word2idx["<start>"]
    end_id = vocab.word2idx["<end>"]
    pad_id = vocab.word2idx["<pad>"]

    # Each beam: (token_ids, cumulative_log_prob, length, alphas_list, distributions)
    # token_ids starts with [<start>]
    beams = [([start_id], 0.0, 0, [], [])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, score, length, alphas, distributions in beams:
            if token_ids and token_ids[-1] == end_id:
                completed.append((token_ids, score, length, alphas, distributions))
                continue

            current_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
            logits, step_alpha = decoder.forward_step(
                features, current_tokens, return_alphas=return_alphas
            )
            step_alpha_cpu = None
            if return_alphas:
                step_alpha_cpu = (
                    step_alpha.squeeze(0).detach().cpu()
                    if step_alpha is not None
                    else torch.zeros(features.size(1))
                )

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            log_probs[start_id] = float("-inf")
            log_probs[pad_id] = float("-inf")

            distribution_values, distribution_indices = log_probs.topk(
                min(token_top_k, log_probs.numel())
            )
            distribution_probabilities = distribution_values.exp().detach().cpu().tolist()
            distribution_ids = distribution_indices.detach().cpu().tolist()
            distribution = [
                (int(word_id), float(probability))
                for probability, word_id in zip(distribution_probabilities, distribution_ids)
            ]
            values, indices = log_probs.topk(min(beam_size, log_probs.numel()))

            for value, word_id in zip(values.tolist(), indices.tolist()):
                candidates.append((
                    token_ids + [word_id],
                    score + value,
                    length + 1,
                    alphas + ([step_alpha_cpu] if return_alphas else []),
                    distributions + [distribution],
                ))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

    final_candidates = completed + beams
    if not final_candidates:
        return ([], torch.empty(0, features.size(1))) if return_alphas else []

    final_candidates.sort(
        key=lambda item: item[1] / length_penalty(item[2], length_penalty_alpha),
        reverse=True,
    )

    if return_nbest:
        results = []
        seen = set()
        for token_ids, score, length, alphas, distributions in final_candidates:
            tokens = token_ids[1:] if token_ids and token_ids[0] == start_id else token_ids[:]
            if end_id in tokens:
                end_index = tokens.index(end_id)
                tokens = tokens[:end_index]
                alphas = alphas[:end_index]
                distributions = distributions[:end_index]
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "token_ids": tokens,
                "beam_score": score / length_penalty(length, length_penalty_alpha),
                "alphas": torch.stack(alphas) if alphas else torch.empty(0, features.size(1)),
                "next_token_distributions": distributions,
            })
            if len(results) >= nbest:
                break
        return results

    best_tokens, _, _, best_alphas, _ = final_candidates[0]

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
