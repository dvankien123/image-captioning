"""Beam-search decoding for one image at a time."""

import torch
import torch.nn.functional as F


def _length_penalty(length, alpha):
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
):
    """Decode one image feature vector and return word IDs (without <start>)."""
    if features.size(0) != 1:
        raise ValueError("beam_search expects features for exactly one image")
    if beam_size < 1 or max_len < 1:
        raise ValueError("beam_size and max_len must be positive")
    if length_penalty_alpha < 0:
        raise ValueError("length_penalty_alpha must be non-negative")

    start_id = vocab.word2idx["<start>"]
    end_id = vocab.word2idx["<end>"]
    pad_id = vocab.word2idx["<pad>"]

    # Training uses the image step to predict <start>; seed search after that step.
    _, initial_state = decoder.lstm(features.unsqueeze(1))
    start_input = decoder.embed(torch.tensor([start_id], device=features.device))

    # Each beam is (IDs, raw log probability, state, next input, emitted length).
    # The seed <start> is context, not a generated caption token.
    beams = [([], 0.0, initial_state, start_input, 0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, score, state, inputs, length in beams:
            # Finished beams are kept separate so they are not expanded again.
            if token_ids and token_ids[-1] == end_id:
                completed.append((token_ids, score, length))
                continue

            # Score the next token from the current word embedding and LSTM state.
            hidden, next_state = decoder.lstm(inputs.unsqueeze(1), state)
            logits = decoder.fc(decoder.dropout(hidden.squeeze(1)))
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            # Special tokens must not be emitted as caption content.
            log_probs[start_id] = float("-inf")
            log_probs[pad_id] = float("-inf")
            values, indices = log_probs.topk(min(beam_size, log_probs.numel()))

            for value, word_id in zip(values.tolist(), indices.tolist()):
                sequence = token_ids + [word_id]
                candidates.append((sequence, score + value, next_state,
                                   decoder.embed(torch.tensor([word_id], device=features.device)), length + 1))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

    # At max_len, score live beams with the same normalization as completed ones.
    final_candidates = completed + [
        (ids, score, length) for ids, score, _, _, length in beams
    ]

    best_ids, _, _ = max(
        final_candidates,
        key=lambda item: item[1] / _length_penalty(item[2], length_penalty_alpha),
    )
    if end_id in best_ids:
        best_ids = best_ids[:best_ids.index(end_id)]
    return best_ids
