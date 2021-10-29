import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import hw_asr.model as module_model
import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

from hw_asr.metric.utils import calc_cer
import jiwer

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    lm_path = ''
    if "lm_path" in config['arch']['args']:
        lm_path = config['arch']['args']['lm_path']
    text_encoder = CTCCharTextEncoder.get_simple_alphabet(lm_path=lm_path)

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    device, device_ids = prepare_device(config["n_gpu"])

    # get function handles of loss and metrics
    loss_fn = config.init_obj(config["loss"], module_loss).to(device)
    metric_fns = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            batch["logits"] = model(**batch)
            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            for j in range(len(batch["text"])):
                argmax_decode = text_encoder.decode(batch["argmax"][j])
                beam_search = text_encoder.beam_search(
                        batch["log_probs"][j], beam_size=100
                    )[:10]
                ground_truth_text = batch["text"][j].lower()
                results.append({
                    "ground_truth": ground_truth_text,
                    "pred_text_argmax": argmax_decode,
                    "pred_text_beam_search": beam_search,
                    "argmax_wer": jiwer.wer(ground_truth_text, argmax_decode) * 100,
                    "argmax_cer": calc_cer(ground_truth_text, argmax_decode) * 100,
                    "bs_wer": jiwer.wer(ground_truth_text, beam_search[0]) * 100,
                    "bs_cer": calc_cer(ground_truth_text, beam_search[0]) * 100
                })
    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default='output.json',
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        required=True,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader"
    )

    config = ConfigParser.from_args(args)
    args = args.parse_args()
    test_data_folder = Path(args.test_data_folder)
    config.config["data"] = {
        "test": {
            "batch_size": args.batch_size,
            "num_workers": args.jobs,
            "datasets": [
                {
                    "type": "CustomDirAudioDataset",
                    "args": {
                        "audio_dir": test_data_folder / "audio",
                        "transcription_dir": test_data_folder / "transcriptions",
			"limit": -1
                    }
                }
            ]
        }
    }
    main(config, args.output)
