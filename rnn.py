import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, ic, hc):
        super(RNNCell, self).__init__()
        self.ic = ic
        self.hc = hc
        self.linear_i = nn.Linear(ic, hc)
        self.linear_h = nn.Linear(hc, hc)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        h = self.tanh(self.linear_i(x) + self.linear_h(h))
        return h


class RNN(nn.Module):
    def __init__(self, ic, hc, layer_num):
        super(RNN, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num

        self.cells = nn.ModuleList([RNNCell(ic, hc)])
        for i in range(self.layer_num - 1):
            cell = RNNCell(hc, hc)
            self.cells.append(cell)

    def forward(self, x, hiddens=None):
        device = x.device
        x_len, bs, _ = x.shape

        if hiddens is None:
            h = torch.zeros(self.layer_num, bs, self.hc).to(device)
        else:
            h = hiddens

        hs = []
        lstm_input = x
        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i]

            for j in range(x_len):
                cur_h = self.cells[i](lstm_input[j], cur_h)
                cur_h = cur_h.unsqueeze(0)
                cur_hs.append(cur_h)

            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)

        out = lstm_input
        hs = torch.cat(hs, dim=0)

        return hs, out
