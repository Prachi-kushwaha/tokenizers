from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozer=True)
class BPETokenizerParams:
    vocab:dict[int, bytes]
    merges:dict[tuple[int, int], int]


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, text:str)->List[int]:
        pass

    @abstractmethod
    def decode(self, indices:List[int])->str:
        pass

def train_bpe(string:str, num_merges:int) -> BPETokenizerParams:

    indices:list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x:bytes([x]) for x in range(256)}

    for i in range(num_merges):
        count = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            count([index1, index2]) += 1

        pair = max(count, key=count.get)
        (index1, index2) = pair

        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merges(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)





