from pathlib import Path

from sklearn.model_selection import train_test_split

import gcNB.tools.train as gcnb_train
import ngRNN.tools.train as ngrnn_train
import gcRNN.tools.train as gcrnn_train


def main(args):
    if args.model_type == "gender-bayes":
        gcnb_train.main(args)
    elif args.model_type == "generate-rnn":
        ngrnn_train.main(args)
    else:
        gcrnn_train.main(args)