"""BLEU metric helpers for image-captioning evaluation."""

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def calculate_bleu(references, hypotheses):
    """Calculate BLEU-1 through BLEU-4 based on input dimensions.

    Sentence input has 2D references (multiple tokenized references) and 1D
    hypotheses (one tokenized caption). Corpus input has 3D references (one
    group of tokenized references per image) and 2D hypotheses.
    """
    if not references or not hypotheses:
        raise ValueError("references and hypotheses must not be empty")

    first_reference = references[0]
    first_hypothesis = hypotheses[0]
    sentence_input = (
        isinstance(first_reference, (list, tuple))
        and isinstance(first_hypothesis, str)
    )
    corpus_input = (
        isinstance(first_reference, (list, tuple))
        and bool(first_reference)
        and isinstance(first_reference[0], (list, tuple))
        and isinstance(first_hypothesis, (list, tuple))
    )
    if not sentence_input and not corpus_input:
        raise ValueError(
            "expected sentence input (2D references, 1D hypotheses) or "
            "corpus input (3D references, 2D hypotheses)"
        )

    if corpus_input:
        if len(references) != len(hypotheses):
            raise ValueError("references and hypotheses must have the same length")
        if any(not image_references for image_references in references):
            raise ValueError("each hypothesis must have at least one reference caption")
        bleu_function = corpus_bleu
    else:
        if not first_reference:
            raise ValueError("references must contain at least one reference caption")
        bleu_function = sentence_bleu

    smoothing = SmoothingFunction().method4
    scores = {}
    for n in range(1, 5):
        weights = (1.0 / n,) * n + (0.0,) * (4 - n)
        scores[f"bleu{n}"] = bleu_function(
            references,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing,
        )
    return scores
