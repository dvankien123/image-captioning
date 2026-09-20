import re
import numpy as np
from gensim.models import Word2Vec

def tokenize(caption):
    caption = caption.lower()
    caption = re.sub(r"([.,!?])", r" \1 ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption.split()
def tokenized_corpus(captions):
    return [tokenize(caption) for caption in captions]
def train_word2vec(captions, embed_dim, window, min_count, sg, epochs, seed, workers):
    model = Word2Vec(
        sentences = tokenized_corpus,
        vector_size = embed_dim,
        window = window,
        min_count = min_count,
        sg = sg,
        epochs = epochs,
        seed = seed,
        workers = workers,
    )
    return model

