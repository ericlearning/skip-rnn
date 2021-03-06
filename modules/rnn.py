import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class RNNCell(nn.Module):
    def __init__(self, ic, hc):
        super(RNNCell, self).__init__()
        self.ic = ic
        self.hc = hc
        self.linear_i = nn.Linear(ic, hc)
        self.linear_h = nn.Linear(hc, hc)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x, h):
        h = self.tanh(self.linear_i(x) + self.linear_h(h))
        return h


class RNN(nn.Module):
    def __init__(self, ic, hc, layer_num, learn_init=False):
        super(RNN, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num

        self.cells = nn.ModuleList([RNNCell(ic, hc)])
        for i in range(self.layer_num - 1):
            cell = RNNCell(hc, hc)
            self.cells.append(cell)
        
        self.hiddens = self.init_hiddens(learn_init)
    
    def init_hiddens(self, learn_init):
        if learn_init:
            h = nn.Parameter(torch.randn(self.layer_num, 1, self.hc))
        else:
            h = nn.Parameters(torch.zeros(self.layer_num, 1, self.hc), requires_grad=False)
        return h

    def forward(self, x, hiddens=None):
        x_len, bs, _ = x.shape

        if hiddens is None:
            h = self.hiddens
        else:
            h = hiddens
        h = h.repeat(1, bs, 1)

        hs = []
        lstm_input = x
        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i]

            for j in range(x_len):
                cur_h = self.cells[i](lstm_input[j], cur_h[0])
                cur_h = cur_h.unsqueeze(0)
                cur_hs.append(cur_h)

            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)

        out = lstm_input
        hs = torch.cat(hs, dim=0)

        return out, (hs,)
