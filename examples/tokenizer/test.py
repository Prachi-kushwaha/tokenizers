from tokenizer import WordTokenizer

tok = WordTokenizer.train("i love myself")
enc = tok.encode("i love")
print(enc)
dec = tok.decode(enc)
print(dec)
