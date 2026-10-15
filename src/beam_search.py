"""Beam-search decoding for one image at a time."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search(decoder, features, vocab, beam_size=3, max_len=30):
    """Decode one image feature vector and return word IDs (without <start>)."""
    if features.size(0) != 1:
        raise ValueError("beam_search expects features for exactly one image")
    if beam_size < 1 or max_len < 1:
        raise ValueError("beam_size and max_len must be positive")

    start_id = vocab.word2idx["<start>"]
    end_id = vocab.word2idx["<end>"]
    pad_id = vocab.word2idx["<pad>"]

    # Training uses the image step to predict <start>; seed search after that step.
    _, initial_state = decoder.lstm(features.unsqueeze(1))
    start_input = decoder.embed(torch.tensor([start_id], device=features.device))

    # Each beam is (generated IDs, log probability, recurrent state, next input(last element of beam)).
    beams = [([], 0.0, initial_state, start_input)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for token_ids, score, state, inputs in beams:
            # Finished beams are kept separate so they are not expanded again.
            if token_ids and token_ids[-1] == end_id:
                completed.append((token_ids, score))
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
                                   decoder.embed(torch.tensor([word_id], device=features.device))))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]

    completed.extend((ids, score) for ids, score, _, _ in beams)
    best_ids, _ = max(completed, key=lambda item: item[1])
    if end_id in best_ids:
        best_ids = best_ids[:best_ids.index(end_id)]
    return best_ids
