from collections import Counter
import nltk
import pickle

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>" : 0, "<start>" : 1, "<end>" : 2, "<unk>" : 3}
        self.idx2word = {0 : "<pad>", 1 : "<start>", 2: "<end>", 3 : "<unk>"}
        self.idx = 4

    def tokenize(self, caption):
        return nltk.word_tokenize(caption.lower())

    def build_vocabulary(self, captions):
        freq_counter = Counter()
        for caption in captions:
            tokenized_caption = self.tokenize(caption)
            freq_counter.update(tokenized_caption)
        for token, freq in freq_counter.items():
            if freq >= self.freq_threshold:
                if token not in self.word2idx:
                    self.word2idx[token] = self.idx
                    self.idx2word[self.idx] = token
                    self.idx += 1
    def numericalize(self, caption):
        tokenized_caption = self.tokenize(caption)
        return [self.word2idx["<start>"]] + [self.word2idx[token] if token in self.word2idx else self.word2idx["<unk>"]
                for token in tokenized_caption] + [self.word2idx["<end>"]]

    def __len__(self):
        return len(self.word2idx)

    def save(self):
        with open("../data/processed/vocab.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load():
        with open("../data/processed/vocab.pkl", "rb") as f:
            return pickle.load(f)



