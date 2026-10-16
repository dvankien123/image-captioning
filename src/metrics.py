"""COCO caption metric helpers for image-captioning evaluation."""

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def _prepare_coco_inputs(references, hypotheses):
    """Validate tokenized corpus inputs and format them for COCO scorers."""
    if not references or not hypotheses:
        raise ValueError("references and hypotheses must not be empty")
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    gts, res = {}, {}
    for index, (image_references, hypothesis) in enumerate(
        zip(references, hypotheses)
    ):
        if not image_references:
            raise ValueError("each hypothesis must have at least one reference caption")
        if isinstance(hypothesis, str) or not isinstance(hypothesis, (list, tuple)):
            raise TypeError("each hypothesis must be a token sequence")
        if any(not isinstance(token, str) for token in hypothesis):
            raise TypeError("hypothesis tokens must be strings")

        reference_texts = []
        for reference in image_references:
            if isinstance(reference, str) or not isinstance(reference, (list, tuple)):
                raise TypeError("each reference caption must be a token sequence")
            if any(not isinstance(token, str) for token in reference):
                raise TypeError("reference tokens must be strings")
            reference_texts.append(" ".join(reference))

        image_id = str(index)
        gts[image_id] = reference_texts
        res[image_id] = [" ".join(hypothesis)]

    return gts, res


def _score(references, hypotheses, scorer):
    gts, res = _prepare_coco_inputs(references, hypotheses)
    score, _ = scorer.compute_score(gts, res)
    return score


def calculate_bleu(references, hypotheses):
    """Calculate corpus BLEU-1 through BLEU-4 using the COCO toolkit.

    ``references`` contains tokenized captions grouped by image, while
    ``hypotheses`` contains one tokenized caption per image.
    """
    gts, res = _prepare_coco_inputs(references, hypotheses)
    scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
    return {
        f"bleu{index + 1}": float(score)
        for index, score in enumerate(scores)
    }


def calculate_meteor(references, hypotheses):
    """Calculate corpus METEOR using the COCO toolkit."""
    scorer = Meteor()
    try:
        score = _score(references, hypotheses, scorer)
        return float(score)
    finally:
        # The toolkit owns a Java subprocess and closes it from its destructor.
        del scorer


def calculate_rouge(references, hypotheses):
    """Calculate corpus ROUGE-L using the COCO toolkit."""
    return float(_score(references, hypotheses, Rouge()))


def calculate_cider(references, hypotheses):
    """Calculate corpus CIDEr using the COCO toolkit."""
    return float(_score(references, hypotheses, Cider()))


def calculate_caption_metrics(references, hypotheses):
    """Calculate all supported corpus metrics for one aligned image batch."""
    gts, res = _prepare_coco_inputs(references, hypotheses)
    bleu_scores, _ = Bleu(4).compute_score(gts, res, verbose=0)
    meteor = Meteor()
    try:
        meteor_score, _ = meteor.compute_score(gts, res)
    finally:
        del meteor
    rouge_score, _ = Rouge().compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)

    scores = {
        f"bleu{index + 1}": float(score)
        for index, score in enumerate(bleu_scores)
    }
    scores["meteor"] = float(meteor_score)
    scores["rouge_l"] = float(rouge_score)
    scores["cider"] = float(cider_score)
    return scores
