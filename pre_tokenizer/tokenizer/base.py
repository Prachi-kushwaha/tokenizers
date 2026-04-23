
from abc import ABC, abstractmethod
from typing import List



class Tokenizer(ABC):

    @abstractmethod
    def train(self, text:str, num_merges:int)->List[int]:
        pass

    @abstractmethod
    def encoder(self, text:str)->List[int]:
        pass

    @abstractmethod
    def decoder(self, indices:List[int])->str:
        pass




