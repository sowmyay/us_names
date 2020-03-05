import argparse
from pathlib import Path

import names.tools.train
import names.tools.predict

parser = argparse.ArgumentParser(prog="names")

subcmd = parser.add_subparsers(dest="command")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter

train = subcmd.add_parser("train", help="Train model", formatter_class=Formatter)
train.add_argument("--data", type=Path, help="Path to directory containing all the US name files")
train.add_argument("--model_type", type=str, choices=("gender-rnn", "gender-bayes", "generate-rnn"),
                   default="gender-bayes")
train.add_argument("--model", type=Path, help="file to save trained model to")
train.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
train.add_argument("--batch-size", type=int, default=64, help="number of chunks per batch")
train.add_argument("--num-epochs", type=int, default=100, help="number of epochs to train for")
train.add_argument("--checkpoint", type=str, required=False, help="path to a model checkpoint (to retrain)")

train.set_defaults(main=names.tools.train.main)


predict = subcmd.add_parser("predict", help="Predict model", formatter_class=Formatter)
predict.add_argument("--model_type", type=str, choices=("gender-rnn", "gender-bayes", "generate-rnn"),
                     default="gender-bayes")
predict.add_argument("--num-workers", type=int, default=0, help="number of parallel workers")
predict.add_argument("--batch-size", type=int, default=64, help="number of chunks per batch")
predict.add_argument("--data", type=Path, help="Path to directory containing all the US name files")
predict.add_argument("--checkpoint", type=Path, required=False, help="path to a model checkpoint")
predict.add_argument("--input", type=str, required=True, help="Make prediction on this input")

predict.set_defaults(main=names.tools.predict.main)


args = parser.parse_args()
args.main(args)

