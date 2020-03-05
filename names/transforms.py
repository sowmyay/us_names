import torch


class JointToTensor:
    def __init__(self, all_states):
        self.all_states = all_states
        self.n_states = len(all_states)

    def __call__(self, state, input, output):
        i = self.all_states.index(state)
        tensor = torch.zeros(1, self.n_states, dtype=torch.long)
        tensor[0][i] = 1
        return tensor
