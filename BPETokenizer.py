from tokenizer import Tokenizer
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import re

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab:dict[int, bytes]
    merges:dict[tuple[int, int], int]

def merge(indices: List[int], pair: tuple[int, int], new_index: int) ->List[int]:
    """ Return updated indices by updating with all instances of `pair` replaced with `new_index`. """
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


def train_bpe(string:str, num_merges:int) -> BPETokenizerParams:

    chunks = re.findall(r"\w+|[^\w\s]", string)

    indices = []
    for chunk in chunks:
        indices.extend(list(chunk.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x:bytes([x]) for x in range(256)}

    for i in range(num_merges):
        count = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            count[(index1, index2)] += 1

        pair = max(count, key=count.get)
        (index1, index2) = pair

        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


