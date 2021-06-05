from common.memory.replay_v2 import ReplayBuffer
import torch
from torch import optim
from torchvision import transforms
import numpy as np
from network import MULTIDQN


class Agent():
    def __init__(self,
                 global_model,
                 lr,
                 channels,
                 n_actions,
                 batch_size,
                 capacity,
                 epsilon,
                 eps_dec,
                 env,
                 device,
                 name,
                 gamma=0.99,
                 replace=1000,
                 img_size=84):
        np.random.seed(1337)
        self.device = device
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.env = env
        self.score = 0
        self.replace = replace
        self.gamma = gamma
        self.batch_size = batch_size
        self.global_model = global_model
        self.dqn = MULTIDQN(lr, channels, n_actions)
        self.loss = torch.nn.MSELoss()
        self.memory = ReplayBuffer(capacity, img_size)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.name = "Agent[" + str(name) + "]"
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((img_size, img_size))])
        self.global_model.to(self.device)
        self.dqn.to(self.device)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            obs = self.transform(obs).unsqueeze(0).to(self.device)
            actions = self.global_model(obs).argmax(1)
            action = actions.detach().item()
        else:
            action = self.env.action_space.sample()

        return action

    def store_experience(self, s, a, r, s_, d):
        self.memory.store_experience(s, a, r, s_, d)

    def update_global_network(self):
        for local_params, global_params in zip(self.dqn.parameters(),
                                               self.global_model.parameters()):
            global_params.grad = local_params

        self.dqn.load_state_dict(self.global_model.state_dict())

    def decrement_eps(self):
        self.epsilon -= self.eps_dec if self.epsilon > 0 else 0

    def update_score(self, reward, best_score):
        self.score += reward

    def learn(self, ):
        if self.memory.mem_cntr < self.batch_size:
            return
        for p in self.global_model.parameters():
            p.grad = None

        indices = np.arange(self.batch_size)

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_ = states_.to(self.device)
        dones = dones.to(self.device)

        q_target = self.global_model(states)[indices, actions]
        q_pred = self.dqn(states_).argmax(dim=1)
        q_pred[dones] = 0.0
        y = rewards + self.gamma * q_pred
        loss = self.loss(y, q_target)
        loss.backward()
        self.optimizer.step()
        self.decrement_eps()
