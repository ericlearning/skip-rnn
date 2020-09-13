import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class GRUCell(nn.Module):
    def __init__(self, ic, hc):
        super(GRUCell, self).__init__()
        self.ic = ic
        self.hc = hc
        self.linear1 = nn.Linear(ic + hc, hc * 2)
        self.linear2 = nn.Linear(ic + hc, hc)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        xavier_normal_(self.linear1.weight)
        xavier_normal_(self.linear2.weight)
        self.linear1.bias.data.fill_(1)
        self.linear2.bias.data.fill_(0)

    def forward(self, x, h):
        out = self.linear1(torch.cat([x, h], 1))
        out = out.split(self.hc, 1)
        r = self.sig(out[0])
        z = self.sig(out[1])
        g = self.tanh(self.linear2(torch.cat([x, r * h], 1)))
        h = z * h + (1 - z) * g
        return h


class GRU(nn.Module):
    def __init__(self, ic, hc, layer_num, learn_init=False):
        super(GRU, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num

        self.cells = nn.ModuleList([GRUCell(ic, hc)])
        for i in range(self.layer_num - 1):
            cell = GRUCell(hc, hc)
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
            cur_h = h[i].unsqueeze(0)

            for j in range(x_len):
                cur_h = self.cells[i](lstm_input[j], cur_h[0])

                # (1, bs, hc)
                cur_h = cur_h.unsqueeze(0)
                cur_hs.append(cur_h)

            # (x_len, bs, hc)
            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)

        out = lstm_input
        hs = torch.cat(hs, dim=0)

        return out, (hs,)
