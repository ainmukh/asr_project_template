import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import os
import numpy as np
from torch import Tensor
import kenlm

from hw_asr.base.base_text_encoder import BaseTextEncoder


class CharTextEncoder(BaseTextEncoder):

    def __init__(self, alphabet: List[str], lm_path: str):
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.lm = None
        if len(lm_path) != 0:
            lm_path = os.path.join(lm_path)
            self.lm = kenlm.LanguageModel(lm_path)

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]) -> str:
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a

    @classmethod
    def get_simple_alphabet(cls, lm_path: str = ''):
        return cls(alphabet=list(ascii_lowercase + ' '), lm_path=lm_path)
