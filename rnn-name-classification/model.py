import torch
import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2o = nn.Linear(in_features=input_size + hidden_size, out_features=output_size)
        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim = 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(size = (1, self.hidden_size))