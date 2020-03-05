import torch

#
# # Find letter index from all_letters, e.g. "a" = 0
# def letterToIndex(letter):
#     return all_letters.find(letter)
#
# # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
# def letterToTensor(letter):
#     tensor = torch.zeros(1, n_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor


class GenderToTensor:
    def __init__(self):
        self.gender = ["male", "female"]

    def __call__(self, g):
        return torch.LongTensor([self.gender.index(g)])


class NameToTensor:
    def __init__(self, all_letters):
        self.all_letters = all_letters
        self.n_letters = len(all_letters) + 1 #EOS

    def __call__(self, name):
        tensor = torch.zeros(len(name), 1, self.n_letters)
        for i, letter in enumerate(name):
            tensor[i][0][self.all_letters.find(letter)] = 1
        return tensor
