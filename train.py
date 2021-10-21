import argparse
import collections
import warnings

import numpy as np
import torch

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.parse_config import ConfigParser
from sklearn.model_selection import train_test_split
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # exit(print(config.__dict__))
    logger = config.get_logger("train")

    # text_encoder
    lm_path = ''
    if "lm_path" in config['arch']['args']:
        lm_path = config['arch']['args']['lm_path']
    text_encoder = CTCCharTextEncoder.get_simple_alphabet(lm_path=lm_path)

    # setup data_loader instances
    if config["data"]["train"]["datasets"][0]["type"] == "LJSpeechDataset":
        df = pd.read_csv("data/datasets/ljspeech/metadata.csv")
        x_train, x_test = train_test_split(df, test_size=0.1, random_state=SEED)
        x_train.to_csv("data/datasets/ljspeech/traindata.csv", sep="|", header=False)
        x_test.to_csv("data/datasets/ljspeech/testdata.csv", sep="|", header=False)
        config["data"]["train"]["datasets"][0]["args"] = "data/datasets/ljspeech/traindata.csv"
        config["data"]["val"]["datasets"][0]["args"] = "data/datasets/ljspeech/testdata.csv"

    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder))
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        text_encoder=text_encoder,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        sr=config["preprocessing"]["sr"]
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
