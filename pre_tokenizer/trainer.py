import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

from trainer_config import getconfig
from datasets import load_dataset
from pathlib import Path
from tokenizer import Tokenizer
from tokenizers.models import Wordlevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path']).mkdir(parents=True, exist_ok=True)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(Wordlevel(unk_token=['UNK']))
        tokenizer.pre_tokenizers = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer




def load_training_dataset(config):
    """
    Production-ready dataset loader
    """

    dataset_name = config["datasource"]

    dataset_config = config.get("dataset_config", None)

    ds = load_dataset(
        path=dataset_name,
        name=dataset_config,
        split='train[:10%]'
    )

    return ds


