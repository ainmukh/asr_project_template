# Don't forget to support cases when target_text == ''
# new imports
import Levenshtein
import jiwer
# from datasets import load_metric
#
#
# cer_metric = load_metric("cer")
# wer_metric = load_metric("wer")


def calc_cer(target_text: str, predicted_text: str) -> float:
    # TODO: your code here
    distance = Levenshtein.distance(target_text, predicted_text)
    return distance / len(target_text)
    # return cer_metric.compute(predicted_text, target_text)
    # raise NotImplementedError()


def calc_wer(target_text: str, predicted_text: str) -> float:
    # TODO: your code here
    return jiwer.wer(target_text, predicted_text)  # / len(target_text.split())
    # return wer_metric.compute(predictions=predicted_text, references=target_text)
    # raise NotImplementedError()


# pred = "i have a dog"
# truth = "i have dogs"
# print('wer =', calc_wer(truth, pred))
