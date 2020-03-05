import torch


class StateToTensor:
    def __init__(self, all_states):
        self.all_states = all_states
        self.n_states = len(all_states)

    def __call__(self, state):
        i = self.all_states.index(state)
        t = torch.zeros(1, self.n_states)
        t[0][i] = 1
        return t


class NameToTensor:
    def __init__(self, all_letters):
        self.all_letters = all_letters
        self.n_letters = len(all_letters) + 1 #EOS

    def __call__(self, name):
        tensor = torch.zeros(len(name), 1, self.n_letters)
        for i, letter in enumerate(name):
            tensor[i][0][self.all_letters.find(letter)] = 1
        return tensor


class TargetToTensor:
    def __init__(self, all_letters):
        self.all_letters = all_letters
        self.n_letters = len(all_letters)

    def __call__(self, name):
        letter_indexes = [self.all_letters.find(letter) for letter in name[1:]]
        letter_indexes.append(self.n_letters)
        return torch.LongTensor(letter_indexes)
