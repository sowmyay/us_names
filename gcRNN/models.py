import torch
import torch.nn as nn

class GCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 2)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch):
        return torch.zeros(batch, 1, self.hidden_size)