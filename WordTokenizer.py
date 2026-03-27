import re
from typing import Dict, List

def build_vocab(string: str) -> Dict[str, int]:
    tokens = re.findall(r"\b\w+\b", string.lower())
    vocab = {word: idx for idx, word in enumerate(set(tokens))}
    return vocab


class WordTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode(self, string: str) -> List[int]:
        tokens = re.findall(r"\b\w+\b", string.lower())
        return [self.vocab.get(word, -1) for word in tokens]  # -1 for unknown

    def decode(self, indices: List[int]) -> str:
        words = [self.inv_vocab.get(i, "<UNK>") for i in indices]
        return " ".join(words)