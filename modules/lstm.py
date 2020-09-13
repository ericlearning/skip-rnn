import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class NaiveLSTMCell(nn.Module):
    def __init__(self, ic, hc):
        super(NaiveLSTMCell, self).__init__()
        self.ic = ic
        self.hc = hc
        self.linears = nn.ModuleDict({
            'if': nn.Linear(ic, hc),
            'hf': nn.Linear(hc, hc),
            'ii': nn.Linear(ic, hc),
            'hi': nn.Linear(hc, hc),
            'ig': nn.Linear(ic, hc),
            'hg': nn.Linear(hc, hc),
            'im': nn.Linear(ic, hc),
            'hm': nn.Linear(hc, hc),
        })
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x, h, c):
        forget_x = self.linears['if'](x) + self.linears['hf'](h)
        forget_gate = self.sig(forget_x)

        input_x = self.linears['ii'](x) + self.linears['hi'](h)
        input_gate = self.sig(input_x)

        output_x = self.linears['ig'](x) + self.linears['hg'](h)
        output_gate = self.sig(output_x)

        mod_x = self.linears['im'](x) + self.linears['hm'](h)
        mod_gate = self.tanh(mod_x)

        c = c * forget_gate + mod_gate * input_gate
        h = output_gate * self.tanh(c)
        return h, c


class LSTMCellOld(nn.Module):
    def __init__(self, ic, hc):
        super(LSTMCellOld, self).__init__()
        self.ic = ic
        self.hc = hc
        self.linear_i = nn.Linear(ic, hc * 4)
        self.linear_h = nn.Linear(hc, hc * 4)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x, h, c):
        out_i, out_h = self.linear_i(x), self.linear_h(h)
        out_i, out_h = out_i.split(self.hc, 1), out_h.split(self.hc, 1)

        forget_gate = self.sig(out_i[0] + out_h[0])
        input_gate = self.sig(out_i[1] + out_h[1])
        output_gate = self.sig(out_i[2] + out_h[2])
        modulation_gate = self.tanh(out_i[3] + out_h[3])

        c = c * forget_gate + modulation_gate * input_gate
        h = output_gate * self.tanh(c)
        return h, c


class LSTMCell(nn.Module):
    def __init__(self, ic, hc, forget_bias=1.0):
        super(LSTMCell, self).__init__()
        self.ic = ic
        self.hc = hc
        self.forget_bias = forget_bias
        self.linear = nn.Linear(ic + hc, hc * 4)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x, h, c):
        out = self.linear(torch.cat([x, h], 1))
        out = out.split(self.hc, 1)

        forget_gate = self.sig(out[0] + self.forget_bias)
        input_gate = self.sig(out[1])
        output_gate = self.sig(out[2])
        modulation_gate = self.tanh(out[3])

        c = c * forget_gate + modulation_gate * input_gate
        h = output_gate * self.tanh(c)
        return h, c


class LSTM(nn.Module):
    def __init__(self, ic, hc, layer_num, learn_init=False):
        super(LSTM, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num

        self.cells = nn.ModuleList([LSTMCell(ic, hc)])
        for i in range(self.layer_num - 1):
            cell = LSTMCell(hc, hc)
            self.cells.append(cell)

        self.h, self.c = self.init_hiddens(learn_init)

    def init_hiddens(self, learn_init):
        if learn_init:
            h = nn.Parameter(torch.randn(self.layer_num, 1, self.hc))
            c = nn.Parameter(torch.randn(self.layer_num, 1, self.hc))
        else:
            h = nn.Parameters(torch.zeros(self.layer_num, 1, self.hc), requires_grad=False)
            c = nn.Parameters(torch.zeros(self.layer_num, 1, self.hc), requires_grad=False)
        return h, c

    def forward(self, x, hiddens=None):
        device = x.device
        x_len, bs, _ = x.shape

        if hiddens is None:
            h, c = self.h, self.c
        else:
            h, c = hiddens
        h, c = h.repeat(1, bs, 1), c.repeat(1, bs, 1)

        hs, cs = [], []
        lstm_input = x
        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i].unsqueeze(0)
            cur_c = c[i].unsqueeze(0)

            for j in range(x_len):
                cur_hiddens = self.cells[i](lstm_input[j], cur_h[0], cur_c[0])
                cur_h, cur_c = cur_hiddens

                # (1, bs, hc) / (1, bs, hc)
                cur_h = cur_h.unsqueeze(0)
                cur_c = cur_c.unsqueeze(0)
                cur_hs.append(cur_h)

            # (x_len, bs, hc)
            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)
            cs.append(cur_c)

        out = lstm_input
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)

        return out, (hs, cs)
