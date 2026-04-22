from .base import Tokenizer
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[Tuple[Tuple[int, int], int]]   # ordered merges


def merge(indices: List[int], pair: Tuple[int, int], new_index: int) -> List[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    @staticmethod
    def train(string: str, num_merges: int) -> BPETokenizerParams:
        pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+"
        chunks = re.findall(pattern, string)

        # faster: no list()
        indices = []
        for chunk in chunks:
            indices.extend(chunk.encode("utf-8"))

        vocab = {x: bytes([x]) for x in range(256)}
        merges = []

        for i in range(num_merges):
            count = defaultdict(int)

            # count pairs
            for a, b in zip(indices, indices[1:]):
                count[(a, b)] += 1

            if not count:
                break

            # get most frequent pair
            pair = max(count, key=count.get)

            new_index = 256 + i
            merges.append((pair, new_index))

            # update vocab
            vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

            # merge indices
            indices = merge(indices, pair, new_index)

        return BPETokenizerParams(vocab=vocab, merges=merges)

    def encode(self, string: str) -> List[int]:
        indices = list(string.encode("utf-8"))

        # apply merges in correct order
        for pair, new_index in self.params.merges:
            indices = merge(indices, pair, new_index)

        return indices

    def decode(self, indices: List[int]) -> str:
        vocab = self.params.vocab
        return b"".join(vocab[i] for i in indices).decode("utf-8")