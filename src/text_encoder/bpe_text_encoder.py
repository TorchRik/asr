import re

import torch
from tokenizers import SentencePieceBPETokenizer

from src.utils.io_utils import ROOT_PATH


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        tokenizer_path = ROOT_PATH / "data" / "tokenizer"
        vocab_path = str(tokenizer_path / "vocab.json")
        merges_txt = str(tokenizer_path / "merges.txt")
        self.bpe_tokenizer = SentencePieceBPETokenizer.from_file(
            vocab_filename=vocab_path,
            merges_filename=merges_txt,
        )
        print()

    def __len__(self):
        return self.bpe_tokenizer.get_vocab_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.tensor:
        text = self.normalize_text(text)
        return torch.tensor(self.bpe_tokenizer.encode(text).ids).unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return self.bpe_tokenizer.decode(inds)

    def ctc_decode(self, inds) -> str:
        # TODO
        return self.decode(inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
