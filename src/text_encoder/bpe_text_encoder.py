import re

import torch
from tokenizers import SentencePieceBPETokenizer

from src.utils.io_utils import ROOT_PATH


class BPETextEncoder:
    EMPTY_TOK = ""

    def __init__(self, save_path, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        tokenizer_path = ROOT_PATH / save_path
        vocab_path = str(tokenizer_path / "vocab.json")
        merges_txt = str(tokenizer_path / "merges.txt")
        self.bpe_tokenizer = SentencePieceBPETokenizer.from_file(
            vocab_filename=vocab_path,
            merges_filename=merges_txt,
        )

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
        # will not implement because of LAS
        return self.decode(inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
