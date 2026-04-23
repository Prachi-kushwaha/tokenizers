from .base import Tokenizer
from collections import defaultdict
import re

class unigram(Tokenizer):
    def __init__(self, vocab):
        self.vocab = vocab

    def train(self, text:str, max_len:4):
        vocab = set()

        for i in range(len(text)):
            for j in range(i+1, min(i+max_len, len(text))+1):
                vocab.add(text[i:j])

        #initialize uniform probabilites
        prob = 1/len(vocab)
        token_prob = {token:prob for token in vocab}
        return token_prob

    # def forward(self, text):
    #     n = len(text)






