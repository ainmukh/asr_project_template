from typing import List, Tuple
import torch
from tqdm import tqdm
import numpy as np

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

from sortedcontainers import SortedSet
from ctcdecoder import beam_search


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], lm_path: str = ''):
        super().__init__(alphabet, lm_path)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def decode(self, inds: List[int]) -> str:
        res = []
        for i in inds:
            cur = self.ind2char[i.item() if torch.is_tensor(i) else i]
            if res and res[-1] == cur or not res and cur == ' ':
                continue
            res.append(cur)
        res = [ch for ch in res if ch != self.EMPTY_TOK]
        return ''.join(res)

    def beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        alphabet = ''.join(self.ind2char.values())
        probs = probs.exp()
        probs = probs.cpu().detach().numpy()
        # hypos = [(1., '', '^')]
        # for i in tqdm(range(probs.shape[0])):
        #     heap = SortedSet([])
        #     hypos_set = {}
        #     while hypos:
        #         prob, hypo, path = hypos.pop()
        #         for j, p in enumerate(probs[i]):
        #             cur_path = self.ind2char[j]
        #             cur_prob = prob * p
        #             cur_hypo = hypo if (path == cur_path or j == 0) else hypo + cur_path
        #             # LM fusion
        #             if self.lm:
        #                 lm_prob = self.lm.score(cur_hypo)
        #                 cur_prob += 0.9 * np.exp(lm_prob) + 0.0001 * i
        #             if (cur_hypo, cur_path) in hypos_set:
        #                 prev_prob = hypos_set[(cur_hypo, cur_path)]
        #                 heap.remove((prev_prob, cur_hypo, cur_path))
        #                 heap.add((cur_prob + prev_prob, cur_hypo, cur_path))
        #                 hypos_set[(cur_hypo, cur_path)] = cur_prob + prev_prob
        #             elif (cur_hypo, cur_path) not in hypos_set:
        #                 if len(heap) < beam_size:
        #                     heap.add((cur_prob, cur_hypo, cur_path))
        #                     hypos_set[(cur_hypo, cur_path)] = cur_prob
        #                 elif cur_prob > heap[0][0]:
        #                     popped = heap.pop(0)
        #                     del hypos_set[(popped[1], popped[2])]
        #                     heap.add((cur_prob, cur_hypo, cur_path))
        #                     hypos_set[(cur_hypo, cur_path)] = cur_prob
        #     hypos = heap
        res = beam_search(probs, alphabet, beam_size=beam_size, lm_model=self.lm, lm_alpha=0.2, lm_beta=0.0001)[0]
        return res

