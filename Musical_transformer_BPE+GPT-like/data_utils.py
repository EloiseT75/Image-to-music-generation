from datasets import load_dataset
import random


def load_abc_dataset(max_examples=None):
    """
    Load ABC notation texts from the Hugging Face dataset
    'sander-wood/irishman'.

    Returns a list of ABC strings.
    """

    dataset = load_dataset("sander-wood/irishman", split="train")

    texts = []
    for example in dataset:
        abc = example.get("abc notation", "")

        if isinstance(abc, str) and abc.strip():
            texts.append(abc.strip())

    if max_examples is not None:
        texts = texts[:max_examples]

    return texts


def train_val_split(texts, val_ratio=0.1, seed=42):
    """
    Split texts into train and validation sets.
    """

    texts = texts.copy()
    random.Random(seed).shuffle(texts)

    n_val = int(len(texts) * val_ratio)

    val_texts = texts[:n_val]
    train_texts = texts[n_val:]

    return train_texts, val_texts