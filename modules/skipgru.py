import torch
import torch.nn as nn
from .gru import GRUCell
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


class SkipGRUCell(nn.Module):
    def __init__(self, ic, hc):
        super(SkipGRUCell, self).__init__()
        self.ste = STELayer()
        self.cell = GRUCell(ic, hc)
        self.linear = nn.Linear(hc, 1)

        xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(1)

    def forward(self, x, u, h, skip=False, delta_u=None):
        # x: (bs, ic)
        # u: (bs, 1)
        # h: (bs, hc)
        # skip: [False or True] * bs
        # delta_u: [skip=True -> (1) / skip=False -> None] * bs

        bs = x.shape[0]
        binarized_u = self.ste(u)                # (bs, 1)

        skip_idx = [i for i, cur_skip in enumerate(skip) if cur_skip]
        skip_num = len(skip_idx)
        no_skip = [not cur_skip for cur_skip in skip]

        if skip_num > 0:
            # (skip_num, ic), (skip_num, 1), (skip_num, hc)
            x_s, u_s, h_s = x[skip], u[skip], h[skip]
            binarized_u_s = binarized_u[skip]        # (skip_num, 1)

            # (skip_num, 1)
            delta_u_s = [cur_delta_u for cur_skip,
                         cur_delta_u in zip(skip, delta_u) if cur_skip]
            delta_u_s = torch.stack(delta_u_s)

            # computing skipped parts
            new_h_s = h_s * (1 - binarized_u_s)        # (skip_num, hc)
            new_u_s = torch.clamp(u_s + delta_u_s, 0, 1) * \
                (1 - binarized_u_s)  # (skip_num, 1)

        if skip_num < bs:
            # (bs-skip_num, ic), (bs-skip_num, 1), (bs-skip_num, hc)
            x_n, u_n, h_n = x[no_skip], u[no_skip], h[no_skip]
            binarized_u_n = binarized_u[no_skip]  # (bs-skip_num, 1)

            # computing non-skipped parts
            new_h_n = self.cell(x_n, h_n)  # (bs-skip_num, hc)
            new_h_n = new_h_n * binarized_u_n            # (bs-skip_num, hc)
            delta_u_n = torch.sigmoid(self.linear(new_h_n))        # (bs-skip_num, 1)
            new_u_n = delta_u_n * binarized_u_n                    # (bs-skip_num, 1)

        # merging skipped and non-skipped parts back
        if 0 < skip_num < bs:
            idx = torch.full((bs,), -1, dtype=torch.long)
            idx[skip_idx] = torch.arange(0, len(skip_idx), dtype=torch.long)
            idx[idx==-1] = torch.arange(len(skip_idx), bs, dtype=torch.long)

            new_u = torch.cat([new_u_s, new_u_n], 0)[idx]        # (bs, 1)
            new_h = torch.cat([new_h_s, new_h_n], 0)[idx]        # (bs, hc)
            delta_u = torch.cat([delta_u_s, delta_u_n], 0)[idx]    # (bs, 1)

        # no need to merge when skip doesn't exist
        elif skip_num == 0:
            new_u = new_u_n
            new_h = new_h_n
            delta_u = delta_u_n

        # no need to merge when everything is skip
        elif skip_num == bs:
            new_u = new_u_s
            new_h = new_h_s
            delta_u = delta_u_s

        n_skips_after = (0.5 / new_u).ceil() - 1  # (bs, 1)
        return binarized_u, new_u, (new_h,), delta_u, n_skips_after


class SkipGRUCellNoSkip(nn.Module):
    def __init__(self, ic, hc):
        super(SkipGRUCellNoSkip, self).__init__()
        self.ste = STELayer()
        self.cell = GRUCell(ic, hc)
        self.linear = nn.Linear(hc, 1)
        
        xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(1)

    def forward(self, x, u, h):
        # x: (bs, ic)
        # u: (bs, 1)
        # h: (bs, hc)

        # computing the states
        binarized_u = self.ste(u)                # (bs, 1)
        new_h = self.cell(x, h)  # (bs, hc)
        new_h = new_h * binarized_u + (1 - binarized_u) * h      # (bs, hc)
        delta_u = torch.sigmoid(self.linear(new_h))        # (bs, 1)
        new_u = delta_u * binarized_u + \
            torch.clamp(u + delta_u, 0, 1) * (1 - binarized_u)  # (bs, 1)

        return binarized_u, new_u, new_h, delta_u


class SkipGRU(nn.Module):
    def __init__(self, ic, hc, layer_num, return_total_u=False, learn_init=False, no_skip=False):
        super(SkipGRU, self).__init__()
        self.ic = ic
        self.hc = hc
        self.layer_num = layer_num
        self.return_total_u = return_total_u
        self.no_skip = no_skip

        if no_skip:
            cur_cell = SkipGRUCellNoSkip
        else:
            cur_cell = SkipGRUCell

        self.cells = nn.ModuleList([cur_cell(ic, hc)])
        for _ in range(self.layer_num - 1):
            cell = cur_cell(hc, hc)
            self.cells.append(cell)

        self.hiddens = self.init_hiddens(learn_init)

    def init_hiddens(self, learn_init):
        if learn_init:
            h = nn.Parameter(torch.randn(self.layer_num, 1, self.hc))
        else:
            h = nn.Parameters(torch.zeros(self.layer_num, 1, self.hc), requires_grad=False)
        return h

    def forward(self, x, hiddens=None):
        device = x.device
        x_len, bs, _ = x.shape    # (x_len, bs, ic)

        if hiddens is None:
            h = self.hiddens
        else:
            h = hiddens
        h = h.repeat(1, bs, 1)
        u = torch.ones(self.layer_num, bs, 1).to(device)            # (l, bs, 1)

        hs = []
        lstm_input = x             # (x_len, bs, ic)

        skip = [False] * bs
        delta_u = [None] * bs

        binarized_us = []

        for i in range(self.layer_num):
            cur_hs = []
            cur_h = h[i].unsqueeze(0)  # (1, bs, hc)
            cur_u = u[i]               # (bs, 1)

            for j in range(x_len):
                if self.no_skip:
                    # (bs, 1), ((bs, hc), (bs, hc)), (bs, 1), (bs, 1)
                    binarized_u, cur_u, cur_h, delta_u = self.cells[i](
                        lstm_input[j], cur_u, cur_h[0])
                    binarized_us.append(binarized_u)
                else:
                    # (bs, 1), ((bs, hc), (bs, hc)), (bs, 1), (bs, 1)
                    binarized_u, cur_u, cur_h, delta_u, n_skips_after = self.cells[i](
                        lstm_input[j], cur_u, cur_h[0], skip, delta_u)
                    binarized_us.append(binarized_u)
                    skip = (n_skips_after[:, 0] > 0).tolist()

                # (1, bs, hc) / (1, bs, hc)
                cur_h = cur_h.unsqueeze(0)
                cur_hs.append(cur_h)

            # (x_len, bs, hc)
            lstm_input = torch.cat(cur_hs, dim=0)
            hs.append(cur_h)

        # (bs, seq * layer_num)
        total_u = torch.cat(binarized_us, 1)
        # (x_len, bs, hc)
        out = lstm_input
        # (l, bs, hc)
        hs = torch.cat(hs, dim=0)

        if self.return_total_u:
            return out, (hs,), total_u
        return out, (hs,)
