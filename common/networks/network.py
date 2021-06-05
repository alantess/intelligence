import os
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch import nn, optim


class NETv1(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 fc1_dims=512,
                 fc2_dims=256,
                 network_name='agent',
                 chkpt_dir='models/'):
        super(NETv1, self).__init__()
        self.file = os.path.join(chkpt_dir, network_name)
        self.convs = nn.Sequential(nn.Conv2d(input_dims, 32, 8, 4), nn.ELU(),
                                   nn.Conv2d(32, 64, 4, 2), nn.ELU(),
                                   nn.Conv2d(64, 64, 3, 1), nn.ELU())
        self.fc1 = nn.Linear(3136, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.track_loss = 0
        self.out = nn.Linear(fc2_dims, n_actions)

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        return q

    def save(self):
        if not os.path.exists(self.chkpt):
            os.mkdir(self.chkpt)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


class NETv0(nn.Module):
    def __init__(self,
                 lr,
                 n_actions,
                 network_name,
                 fc1_dims=256,
                 fc2_dims=256,
                 chkpt_dir='models'):
        super(NETv0, self).__init__()
        self.chkpt = chkpt_dir
        self.file = os.path.join(chkpt_dir, network_name + '_ddqn.pt')
        self.candle_conv = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
                                         nn.MaxPool2d(2),
                                         nn.Conv2d(32, 64, 4, 4), nn.ReLU(),
                                         nn.MaxPool2d(2))
        self.gaf_conv = nn.Sequential(nn.Conv2d(7, 32, 8, 4), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 2),
                                      nn.ReLU())

        self.fc1 = nn.Linear(640, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.out = nn.Linear(fc2_dims, n_actions)
        self.normalize = transforms.Compose([
            transforms.Resize(84),
            transforms.Normalize((0.485, 0.456, 0.406, 0.400),
                                 (0.229, 0.224, 0.225, 0.220))
        ])
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, candle, gaf):
        candle = self.normalize(candle)
        gaf = self.gaf_conv(gaf)
        candle = self.candle_conv(candle)
        x = torch.cat([candle.flatten(1), gaf.flatten(1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        return q

    def save(self):
        if not os.path.exists(self.chkpt):
            os.mkdir(self.chkpt)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
