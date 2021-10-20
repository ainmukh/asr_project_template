from hw_asr.base.base_text_encoder import BaseTextEncoder
import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import os
import numpy as np
from torch import Tensor
import kenlm
import torch

import re
from tqdm.auto import tqdm
import youtokentome as yttm
from sortedcontainers import SortedSet


class BytePairEncoder(BaseTextEncoder):

    def __init__(self, trained_tokenizer_path: str, lm_path: str):
        self.encoder = yttm.BPE(model=trained_tokenizer_path)
        self.lm = None
        if len(lm_path) != 0:
            lm_path = os.path.join(lm_path)
            self.lm = kenlm.LanguageModel(lm_path)

    def __len__(self):
        return self.encoder.vocab_size()

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.encoder.id_to_subword(item)

    def encode(self, text: str) -> Tensor:
        text = self.normalize_text(text)
        return Tensor(self.encoder.encode(text)).to(torch.int)

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]) -> str:
        if vector is not List:
            return self.encoder.decode(vector.tolist())
        else:
            return self.encoder.decode(vector)

    def beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == self.__len__()

        probs = probs.cpu().detach().numpy()
        hypos = [(1., '', '')]

        for i in tqdm(range(probs.shape[0])):
            heap = SortedSet([])
            hypos_set = {}
            while hypos:
                prob, hypo, path = hypos.pop()
                for j, p in enumerate(probs[i]):
                    cur_path = self.encoder.id_to_subword(j)
                    cur_prob = prob * p
                    cur_hypo = hypo if path == cur_path else hypo + cur_path
                    # LM fusion
                    if self.lm:
                        lm_prob = self.lm(cur_hypo)
                        cur_prob += 0.9 * np.exp(lm_prob) + 0.0001 * i
                    if (cur_hypo, cur_path) in hypos_set:
                        prev_prob = hypos_set[(cur_hypo, cur_path)]
                        heap.remove((prev_prob, cur_hypo, cur_path))
                        heap.add((cur_prob + prev_prob, cur_hypo, cur_path))
                        hypos_set[(cur_hypo, cur_path)] = cur_prob + prev_prob
                    elif (cur_hypo, cur_path) not in hypos_set:
                        if len(heap) < beam_size:
                            heap.add((cur_prob, cur_hypo, cur_path))
                            hypos_set[(cur_hypo, cur_path)] = cur_prob
                        elif cur_prob > heap[0][0]:
                            popped = heap.pop(0)
                            del hypos_set[(popped[1], popped[2])]
                            heap.add((cur_prob, cur_hypo, cur_path))
                            hypos_set[(cur_hypo, cur_path)] = cur_prob

            hypos = heap
        p, res, path = hypos[-1]
        return [(res, 0. if p is None else p)]

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
