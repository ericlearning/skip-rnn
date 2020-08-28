import math
import torch
import torch.nn as nn
from .lstm import LSTMCell
from torch.autograd import Function
from torch.nn.init import xavier_normal_


class STEFunction(Function):
    @staticmethod
    def forward(self, x):
        return x.round()

    @staticmethod
    def backward(self, grad):
        return grad


class STELayer(nn.Module):
    def __init__(self):
        super(STELayer, self).__init__()

    def forward(self, x):
        binarizer = STEFunction.apply
        return binarizer(x)


class SkipLSTMCellBasic(nn.Module):
        # This SkipLSTMCell only works when bs=1
    def __init__(self, ic, hc):
        super(SkipLSTMCellBasic, self).__init__()
        self.ste = STELayer()
        self.cell = LSTMCell(ic, hc)
        self.linear = nn.Linear(hc, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(1)

    def forward(self, x, u, h, c, skip=False, delta_u=None):
        # x: (bs, ic)
        # u: (bs, 1)
        # h: (bs, hc)
        # c: (bs, hc)
        # skip: False or True
        # delta_u: skip=True -> (bs, 1) / skip=False -> None

        binarized_u = self.ste(u)                # (bs, 1)

        if skip:
            new_h = h * (1 - binarized_u)        # (bs, hc)
            new_c = c * (1 - binarized_u)        # (bs, hc)
            new_u = torch.clamp(u + delta_u, 0, 1) * \
                (1 - binarized_u)  # (bs, 1)
        else:
            new_h, new_c = self.cell(x, h, c)  # (bs, hc), (bs, hc)
            new_h = new_h * binarized_u            # (bs, hc)
            new_c = new_c * binarized_u            # (bs, hc)
            delta_u = torch.sigmoid(self.linear(new_c))        # (bs, 1)
            new_u = delta_u * binarized_u                    # (bs, 1)

        n_skips_after = (0.5 / new_u).ceil() - 1  # (bs, 1)
        return new_u, (new_h, new_c), delta_u, n_skips_after


class SkipLSTMCell(nn.Module):
    def __init__(self, ic, hc):
        super(SkipLSTMCell, self).__init__()
        self.ste = STELayer()
        self.cell = LSTMCell(ic, hc)
        self.linear = nn.Linear(hc, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(1)

    def forward(self, x, u, h, c, skip=False, delta_u=None):
        # x: (bs, ic)
        # u: (bs, 1)
        # h: (bs, hc)
        # c: (bs, hc)
        # skip: [False or True] * bs
        # delta_u: [skip=True -> (1) / skip=False -> None] * bs

        bs = x.shape[0]
        binarized_u = self.ste(u)                # (bs, 1)

        skip_idx = [i for i, cur_skip in enumerate(skip) if cur_skip]
        skip_num = len(skip_idx)
        no_skip = [not cur_skip for cur_skip in skip]

        if skip_num > 0:
            # (skip_num, ic), (skip_num, 1), (skip_num, hc), (skip_num, hc)
            x_s, u_s, h_s, c_s = x[skip], u[skip], h[skip], c[skip]
            binarized_u_s = binarized_u[skip]        # (skip_num, 1)

            # (skip_num, 1)
            delta_u_s = [cur_delta_u for cur_skip,
                         cur_delta_u in zip(skip, delta_u) if cur_skip]
            delta_u_s = torch.stack(delta_u_s)

            # computing skipped parts
            new_h_s = h_s * (1 - binarized_u_s)        # (skip_num, hc)
            new_c_s = c_s * (1 - binarized_u_s)        # (skip_num, hc)
            new_u_s = torch.clamp(u_s + delta_u_s, 0, 1) * \
                (1 - binarized_u_s)  # (skip_num, 1)

        if skip_num < bs:
            # (bs-skip_num, ic), (bs-skip_num, 1), (bs-skip_num, hc), (bs-skip_num, hc)
            x_n, u_n, h_n, c_n = x[no_skip], u[no_skip], h[no_skip], c[no_skip]
            binarized_u_n = binarized_u[no_skip]  # (bs-skip_num, 1)

            # computing non-skipped parts
            new_h_n, new_c_n = self.cell(x_n, h_n, c_n)  # (bs-skip_num, hc), (bs-skip_num, hc)
            new_h_n = new_h_n * binarized_u_n            # (bs-skip_num, hc)
            new_c_n = new_c_n * binarized_u_n            # (bs-skip_num, hc)
            delta_u_n = torch.sigmoid(self.linear(new_c_n))        # (bs-skip_num, 1)
            new_u_n = delta_u_n * binarized_u_n                    # (bs-skip_num, 1)

        # merging skipped and non-skipped parts back
        if 0 < skip_num < bs:
            idx = torch.full((bs,), -1, dtype=torch.long)
            idx[skip_idx] = torch.arange(0, len(skip_idx), dtype=torch.long)
            idx[idx==-1] = torch.arange(len(skip_idx), bs, dtype=torch.long)

            new_u = torch.cat([new_u_s, new_u_n], 0)[idx]        # (bs, 1)
            new_h = torch.cat([new_h_s, new_h_n], 0)[idx]        # (bs, hc)
            new_c = torch.cat([new_c_s, new_c_n], 0)[idx]        # (bs, hc)
            delta_u = torch.cat([delta_u_s, delta_u_n], 0)[idx]    # (bs, 1)

        # no need to merge when skip doesn't exist
        elif skip_num == 0:
            new_u = new_u_n
            new_h = new_h_n
            new_c = new_c_n
            delta_u = delta_u_n

        # no need to merge when everything is skip
        elif skip_num == bs:
            new_u = new_u_s
            new_h = new_h_s
            new_c = new_c_s
            delta_u = delta_u_s

        n_skips_after = (0.5 / new_u).ceil() - 1  # (bs, 1)
        return new_u, (new_h, new_c), delta_u, n_skips_after


class SkipLSTMCellNoSkip(nn.Module):
    def __init__(self, ic, hc):
        super(SkipLSTMCellNoSkip, self).__init__()
        self.ste = STELayer()
        self.cell = LSTMCell(ic, hc)
        self.linear = nn.Linear(hc, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.fill_(1)

    def forward(self, x, u, h, c):
        # x: (bs, ic)
        # u: (bs, 1)
        # h: (bs, hc)
        # c: (bs, hc)

        # computing the states
        binarized_u = self.ste(u)                # (bs, 1)
        new_h, new_c = self.cell(x, h, c)  # (bs, hc), (bs, hc)
        new_h = new_h * binarized_u + (1 - binarized_u) * h      # (bs, hc)
        new_c = new_c * binarized_u + (1 - binarized_u) * c      # (bs, hc)
        delta_u = torch.sigmoid(self.linear(new_c))        # (bs, 1)
        new_u = delta_u * binarized_u + \
            torch.clamp(u + delta_u, 0, 1) * (1 - binarized_u)  # (bs, 1)

        return new_u, (new_h, new_c), delta_u


class SkipLSTM(nn.Module):
    def __init__(self, ic, hc, layer_num, return_total_u=False, learn_init=False, no_skip=False):
        super(SkipLSTM, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num
        self.return_total_u = return_total_u
        self.learn_init = learn_init
        self.no_skip = no_skip

        if no_skip:
            cur_cell = SkipLSTMCellNoSkip
        else:
            cur_cell = SkipLSTMCell

        self.cells = nn.ModuleList([cur_cell(ic, hc)])
        for _ in range(self.layer_num - 1):
            cell = cur_cell(hc, hc)
            self.cells.append(cell)

    def forward(self, x, hiddens=None):
        device = x.device
        x_len, bs, _ = x.shape    # (x_len, bs, ic)

        if hiddens is None:
            if self.learn_init:
                h = nn.Parameter(torch.randn(self.layer_num, bs, self.hc))
                c = nn.Parameter(torch.randn(self.layer_num, bs, self.hc))
            else:
                h = torch.zeros(self.layer_num, bs, self.hc)
                c = torch.zeros(self.layer_num, bs, self.hc)
            h, c = h.to(device), c.to(device)
            u = torch.ones(self.layer_num, bs, 1).to(device)            # (l, bs, 1)
        else:
            h, c, u = hiddens

        hs, cs = [], []
        lstm_input = x             # (x_len, bs, ic)

        skip = [False] * bs
        delta_u = [None] * bs

        total_u = 0

        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i].unsqueeze(0)  # (1, bs, hc)
            cur_c = c[i].unsqueeze(0)  # (1, bs, hc)
            cur_u = u[i]               # (bs, 1)

            for j in range(x_len):
                if self.no_skip:
                    # (bs, 1), ((bs, hc), (bs, hc)), (bs, 1), (bs, 1)
                    cur_u, cur_hiddens, delta_u = self.cells[i](
                        lstm_input[j], cur_u, cur_h[0], cur_c[0])
                    total_u += cur_u.mean()
                    # (bs, hc), (bs, hc)
                    cur_h, cur_c = cur_hiddens
                else:
                    # (bs, 1), ((bs, hc), (bs, hc)), (bs, 1), (bs, 1)
                    cur_u, cur_hiddens, delta_u, n_skips_after = self.cells[i](
                        lstm_input[j], cur_u, cur_h[0], cur_c[0], skip, delta_u)
                    total_u += cur_u.mean()
                    # (bs, hc), (bs, hc)
                    cur_h, cur_c = cur_hiddens
                    skip = (n_skips_after[:, 0] > 0).tolist()

                # (1, bs, hc) / (1, bs, hc)
                cur_h = cur_h.unsqueeze(0)
                cur_c = cur_c.unsqueeze(0)
                cur_hs.append(cur_h)

            # (x_len, bs, hc)
            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)
            cs.append(cur_c)

        total_u /= (self.layer_num * x_len)
        # (x_len, bs, hc)
        out = lstm_input
        # (l, bs, hc)
        hs = torch.cat(hs, dim=0)
        # (l, bs, hc)
        cs = torch.cat(cs, dim=0)

        if self.return_total_u:
            return out, (hs, cs), total_u
        return out, (hs, cs)
