import torch 
import torch.nn as nn
import torch.nn.functional as F


class PRnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, x):
        '''x: (B, T, D) '''
        hidden, _ = self.rnn(x)
        hidden, _ = self.attn(hidden, hidden, hidden)
        hidden = hidden[:, -1]
        y = self.classifier(hidden)
        y = F.softmax(y,dim=-1)
        return y