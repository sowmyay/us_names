import string

import torch

from ngRNN.transforms import (
    StateToTensor,
    NameToTensor
)
from ngRNN.models import RNN

max_length = 20


def predict(rnn, state, all_states, all_letters, n_letters, start_letter='A'):
    with torch.no_grad():

        state_transform = StateToTensor(all_states)
        name_transform = NameToTensor(all_letters)

        category_tensor = state_transform(state)
        input = name_transform(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = name_transform(letter)

        return output_name


def get_all_states(datapath):
    states = []
    for path in datapath.glob('*.TXT'):
        states += [path.stem]
    return states


def main(args):
    def map_location(storage, _):
        return storage.cuda() if torch.cuda.is_available() else storage.cpu()

    all_states = get_all_states(args.data)
    n_states = len(all_states)
    all_letters = string.ascii_letters
    n_letters = len(all_letters) + 1
    rnn = RNN(n_letters, 128, n_letters, n_states)

    chkpt = torch.load(args.checkpoint, map_location=map_location)
    rnn.load_state_dict(chkpt["state_dict"])

    return predict(rnn, args.state, all_states, all_letters, n_letters, args.start_letter)
