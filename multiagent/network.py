import os
import torch
from torch import nn, optim
import torch.nn.functional as F


class SharedAdam(optim.Adam):
    def __init__(self,
                 params,
                 lr,
                 betas=(0.9, 0.99),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params,
                                         lr=lr,
                                         betas=betas,
                                         eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0

                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(
                    p, memory_format=torch.preserve_format)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class MULTIDQN(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 fc1_dims=512,
                 fc2_dims=256,
                 network_name='agent',
                 chkpt_dir='models/'):
        super(MULTIDQN, self).__init__()
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