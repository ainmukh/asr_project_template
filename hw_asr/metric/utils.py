import Levenshtein
import jiwer


def calc_cer(target_text: str, predicted_text: str) -> float:
    distance = Levenshtein.distance(target_text, predicted_text)
    return 1.0 if len(target_text) == 0 else distance / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    return jiwer.wer(target_text, predicted_text)
