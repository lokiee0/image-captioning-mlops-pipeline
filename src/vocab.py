"""Simple whitespace-tokenizer vocabulary, dataset-agnostic."""
import json
import re
from collections import Counter

SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


class Vocabulary:
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    @staticmethod
    def tokenize(text: str):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return text.split()

    def build(self, captions: list[str]):
        counter = Counter()
        for c in captions:
            counter.update(self.tokenize(c))

        words = [w for w, freq in counter.items() if freq >= self.min_freq]
        vocab = SPECIAL_TOKENS + sorted(words)
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, text: str, max_len: int):
        tokens = ["<start>"] + self.tokenize(text)[: max_len - 2] + ["<end>"]
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: list[int]):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), "<unk>")
            if w == "<end>":
                break
            if w not in ("<start>", "<pad>"):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str, min_freq: int = 3):
        v = cls(min_freq=min_freq)
        with open(path) as f:
            v.word2idx = json.load(f)
        v.idx2word = {i: w for w, i in v.word2idx.items()}
        return v
