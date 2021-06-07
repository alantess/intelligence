import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch import nn, optim


class QVectNet(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 fc1_dims=256,
                 fc2_dims=256,
                 network_name='agent.pt',
                 chkpt_dir='models/'):
        super(QVectNet, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.file = os.path.join(chkpt_dir, network_name)
        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.out = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def save(self):
        if not os.path.exists(self.chkpt):
            os.mkdir(self.chkpt)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))


class QConvNet(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 fc1_dims=512,
                 fc2_dims=256,
                 network_name='agent',
                 chkpt_dir='models/'):
        super(QConvNet, self).__init__()
        self.chkpt = chkpt_dir
        self.file = os.path.join(chkpt_dir, network_name)
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.track_loss = 0
        self.out = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = F.relu(self.fc2(flat1))
        actions = self.out(actions)

        return actions

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def save(self):
        if not os.path.exists(self.chkpt):
            os.mkdir(self.chkpt)
        torch.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(torch.load(self.file))
