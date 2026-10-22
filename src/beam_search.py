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
    """Decode one image feature map and return word IDs (without <start>)."""
    if features.size(0) != 1:
        raise ValueError("beam_search expects features for exactly one image")
    if beam_size < 1 or max_len < 1:
        raise ValueError("beam_size and max_len must be positive")
    if length_penalty_alpha < 0:
        raise ValueError("length_penalty_alpha must be non-negative")

    start_id = vocab.word2idx["<start>"]
    end_id = vocab.word2idx["<end>"]
    pad_id = vocab.word2idx["<pad>"]

    h, c = decoder.init_hidden_state(features)
    # Each beam: (token_ids, score, (h, c), prev_word_id, length, alphas)
    beams = [([], 0.0, (h, c), start_id, 0, [])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, score, (h_prev, c_prev), prev_word_id, length, alphas in beams:
            if token_ids and token_ids[-1] == end_id:
                completed.append((token_ids, score, length, alphas))
                continue

            context, alpha = decoder.attention(features, h_prev)
            word_emb = decoder.embed(torch.tensor([prev_word_id], device=features.device))
            lstm_input = torch.cat([word_emb, context], dim=1)
            h_next, c_next = decoder.lstm(lstm_input, (h_prev, c_prev))

            logits = decoder.fc(decoder.dropout(h_next))
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            log_probs[start_id] = float("-inf")
            log_probs[pad_id] = float("-inf")
            values, indices = log_probs.topk(min(beam_size, log_probs.numel()))

            for value, word_id in zip(values.tolist(), indices.tolist()):
                sequence = token_ids + [word_id]
                step_alpha = alpha.squeeze(0).detach().cpu()
                candidates.append((
                    sequence,
                    score + value,
                    (h_next, c_next),
                    word_id,
                    length + 1,
                    alphas + [step_alpha],
                ))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

    final_candidates = completed + [
        (ids, score, length, alphas) for ids, score, _, _, length, alphas in beams
    ]

    best_ids, _, _, best_alphas = max(
        final_candidates,
        key=lambda item: item[1] / length_penalty(item[2], length_penalty_alpha),
    )
    if end_id in best_ids:
        idx = best_ids.index(end_id)
        best_ids = best_ids[:idx]
        best_alphas = best_alphas[:idx]

    if return_alphas:
        alpha_tensor = torch.stack(best_alphas) if best_alphas else torch.empty(0, features.size(1))
        return best_ids, alpha_tensor
    return best_ids
