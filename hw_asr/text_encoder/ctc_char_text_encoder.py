from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        # raise NotImplementedError()
        # res = []
        # for i in inds:
        #     res.append(self.ind2char[i.item()])
        # return ''.join(res)
        res = []
        for i in inds:
            cur = self.ind2char[i.item() if torch.is_tensor(i) else i]
            if res and res[-1] == cur or not res and cur == ' ':
                continue
            res.append(cur)
            # print(cur, res)
        res = [ch for ch in res if ch != self.EMPTY_TOK]
        res = [res[i] for i in range(len(res)) if i == 0 or res[i] != res[i - 1]]
        return ''.join(res)

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x[1], reverse=True)
