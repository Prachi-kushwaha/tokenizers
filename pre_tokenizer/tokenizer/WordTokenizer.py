import re
from  .base import Tokenizer
from typing import Dict, List

class WordTokenizer(Tokenizer):
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def train(self, text:str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        vocab = {word: idx for idx, word in enumerate(set(tokens))}
        return vocab

    def encoder(self, text:str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [self.vocab.get(word, -1) for word in tokens]

    def decoder(self, indices: List[int]) -> str:
        words = [self.inv_vocab.get(i, "<UNK>") for i in indices]
        return " ".join(words)
