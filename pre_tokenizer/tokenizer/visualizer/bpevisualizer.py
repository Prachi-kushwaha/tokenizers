from BPETokenizer import merge
from BPETokenizer import BPETokenizers


def show_token_mapping(tokenizer, encoded):

    for token in encoded:
        piece = tokenizer.params.vocab[token].decode("utf-8", errors="replace")
        print(f"{token}->{piece}")

def show_bpe_steps(tokenizer: BPETokenizers, text: str):
    vocab = tokenizer.params.vocab
    merges = tokenizer.params.merges

    # start from raw bytes
    indices = list(text.encode("utf-8"))

    def decode(indices):
        return "".join(vocab[i].decode("utf-8", errors="replace") for i in indices)

    # print("Initial indices:", indices)
    # print("Initial text:", decode(indices))

    for step, (pair, new_index) in enumerate(merges, 1):
        pair_str = (
            vocab[pair[0]].decode("utf-8", errors="replace"),
            vocab[pair[1]].decode("utf-8", errors="replace")
        )

        print(f"\n Step {step} ")
        print(f"Merging pair: {pair} -> {pair_str}")
        print(f"New token id: {new_index}")
        print(f"Represents: '{vocab[new_index].decode('utf-8', errors='replace')}'")

        indices = merge(indices, pair, new_index)

        print("Updated indices:", indices)
        print("Decoded:", decode(indices))
