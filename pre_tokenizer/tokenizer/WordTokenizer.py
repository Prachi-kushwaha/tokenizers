import re
from  .base import Tokenizer
from typing import Dict, List

class WordTokenizer(Tokenizer):
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def train(cls, text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        vocab = {word: idx for idx, word in enumerate(set(tokens))}
        return cls(vocab)

    def encode(self, text):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [self.vocab.get(word, -1) for word in tokens]

    def decode(self):
        token = "hello"