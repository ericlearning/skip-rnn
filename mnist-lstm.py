import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from modules.rnn import RNN
from modules.gru import GRU
from modules.lstm import LSTM

torch.backends.cudnn.benchmark = True

bs = 64
lr = 0.0001
epoch_num = 10
gradient_clip = False
read_size = 1
learn_init = True
use_multigpu = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

record_pth = 'runs/lstm'
os.makedirs(record_pth, exist_ok=True)
writer = SummaryWriter(record_pth)


class Net(nn.Module):
    def __init__(self, mode, learn_init=False):
        super(Net, self).__init__()
        if mode == 'custom_lstm':
            self.m = LSTM(read_size, 110, 1, learn_init=learn_init)
        elif mode == 'custom_gru':
            self.m = GRU(read_size, 110, 1, learn_init=learn_init)
        elif mode == 'custom_rnn':
            self.m = RNN(read_size, 110, 1, learn_init=learn_init)
        elif mode == 'lstm':
            self.m = nn.LSTM(read_size, 110, 1)
        elif mode == 'gru':
            self.m = nn.LSTM(read_size, 110, 1)
        elif mode == 'rnn':
            self.m = nn.RNN(read_size, 110, 1)
        self.fc = nn.Linear(110, 10)

    def forward(self, x):
        # x_len, bs, ic
        x = x.permute(1, 0, 2)
        out = self.m(x)[0][-1, :, :]
        out = self.fc(out)
        return out


net = Net(mode='custom_lstm', learn_init=learn_init).to(device)
if use_multigpu:
    net = nn.DataParallel(net)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_ds = datasets.MNIST(root='data', train=True, transform=tf, download=True)
val_ds = datasets.MNIST(root='data', train=False, transform=tf, download=True)

train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=10)
val_dl = DataLoader(val_ds, bs, shuffle=True, num_workers=10)


opt = optim.Adam(net.parameters(), lr, betas=(0.9, 0.999))
ce = nn.CrossEntropyLoss()


def train(dl, epoch_num):
    net.train()
    dl_len = len(dl)
    for i, (x, t) in enumerate(tqdm(dl)):
        cur_bs = x.shape[0]
        opt.zero_grad()
        x = x.to(device)
        t = t.to(device)
        x = x.reshape(cur_bs, -1, read_size)

        pred = net(x)
        loss = ce(pred, t)
        loss = loss.mean()
        loss.backward()
        if gradient_clip != False:
        	nn.utils.clip_grad_norm_(net.parameters(), gradient_clip)
        opt.step()

        writer.add_scalar('train loss', float(loss), epoch_num * dl_len + i)


@torch.no_grad()
def val(dl, epoch_num):
    net.eval()
    loss_avg = 0
    acc_avg = 0
    for x, t in dl:
        cur_bs = x.shape[0]
        x = x.to(device)
        t = t.to(device)
        x = x.reshape(cur_bs, -1, read_size)

        pred = net(x)
        pred_logit = pred.argmax(-1)
        loss = ce(pred, t)
        loss = loss.mean()
        acc = (pred_logit == t).sum().cpu().numpy() / cur_bs

        loss_avg += loss
        acc_avg += acc

    loss_avg /= len(dl)
    acc_avg /= len(dl)
    writer.add_scalar('val loss', float(loss_avg), epoch_num)
    writer.add_scalar('val acc', float(acc_avg), epoch_num)

    return loss_avg, acc_avg


for i in range(epoch_num):
    train(train_dl, i)
    loss_avg, acc_avg = val(val_dl, i)
    print(f'epoch: {i} / loss: {loss_avg} / acc: {acc_avg}')

writer.close()
