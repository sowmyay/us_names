import string

import torch

from gcRNN.models import GCRNN
from gcRNN.datasets import GCPredictDataset


def predict(input_line, chkpt, n_predictions=1):
    n_letters = len(string.ascii_letters) + 1
    rnn = GCRNN(n_letters, 128, 2)

    def map_location(storage, _):
        return storage.cuda() if torch.cuda.is_available() else storage.cpu()

    chkpt = torch.load(chkpt, map_location=map_location)
    rnn.load_state_dict(chkpt["state_dict"])

    transform = GCPredictDataset()
    input = transform[input_line].unsqueeze(0)

    hidden = rnn.initHidden(1)
    # Get prediction for all samples/lines
    for i in range(input.size(0)):
        output, hidden = rnn(input[:, i], hidden)

    # Get top N categories (probability, category_index)
    output = output.squeeze(0).squeeze(0)

    topv, topi = torch.topk(output, 1)

    # Store prediction result to a patent
    predictions = []
    genders = ["male", "female"]
    print(output)
    print(topv, topi.item())
    print(input_line)
    print(genders[topi.item()])


def main(args):
    predict(args.input, args.checkpoint)
