from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def train_tokenizer(texts, vocab_size=1024, save_path="tokenizer.json"):
    """
    Train a BPE tokenizer on a list of texts.
    """

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    tokenizer.train_from_iterator(texts, trainer)

    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(save_path)

    return tokenizer


def load_tokenizer(path="tokenizer.json"):
    """
    Load a saved tokenizer.
    """

    return Tokenizer.from_file(path)


def encode_text(tokenizer, text):
    """
    Convert text to token ids.
    """

    return tokenizer.encode(text).ids


def decode_tokens(tokenizer, token_ids):
    """
    Convert token ids back to text.
    """

    return tokenizer.decode(token_ids)