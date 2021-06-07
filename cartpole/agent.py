import numpy as np
import torch
from common.memory.replay import ReplayBuffer
from common.networks.network import *


class Agent:
    def __init__(self,
                 lr,
                 input_dims,
                 n_actions,
                 batch_size,
                 capacity,
                 env,
                 device,
                 epsilon=1.0,
                 eps_dec=4.5e-6,
                 gamma=0.99,
                 eps_min=0.01,
                 replace=1000,
                 img_mode=True):
        self.epsilon = epsilon
        self.gamma = gamma
        self.indices = np.arange(batch_size)
        self.replace = replace
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.env = env
        self.img_mode = img_mode
        if self.img_mode:
            print("Image Agent Activated.")
            self.q_eval = QConvNet(lr,
                                   input_dims,
                                   n_actions,
                                   network_name='q_eval_img.pt')
            self.q_next = QConvNet(lr,
                                   input_dims,
                                   n_actions,
                                   network_name='q_next_img.pt')

        else:
            print('Vector Agent Activated.')
            self.q_eval = QVectNet(lr,
                                   input_dims,
                                   n_actions,
                                   network_name='q_eval_vect.pt')
            self.q_next = QVectNet(lr,
                                   input_dims,
                                   n_actions,
                                   network_name='q_next_vect.pt')

        self.memory = ReplayBuffer(capacity, input_dims)
        self.learn_step_cntr = 0
        self.device = device

        self.q_next.to(device)
        self.q_eval.to(device)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            if self.img_mode:
                obs = obs.reshape(3, 210, 160)
            obs = torch.tensor([obs], dtype=torch.float32).to(self.device)

            actions = self.q_eval(obs)
            action = torch.argmax(actions, dim=1).item()
        else:
            action = self.env.action_space.sample()

        return action

    def store_experience(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def experience_to_tensor(self):
        s, a, r, s_, d = self.memory.sample(self.batch_size)
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s_ = torch.tensor(s_).to(self.device)
        d = torch.tensor(d).to(self.device)

        if self.img_mode:
            s = s.permute(0, 3, 1, 2)
            s_ = s_.permute(0, 3, 1, 2)

        return s, a, r, s_, d

    def decrement_eps(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def save(self):
        print('saving...')
        self.q_eval.save()
        self.q_next.save()

    def load(self):
        print('loading...')
        self.q_next.load()
        self.q_eval.load()

    def replace_target_network(self):
        if self.learn_step_cntr % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        for p in self.q_eval.parameters():
            p.grad = None

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.experience_to_tensor()

        q_pred = self.q_eval(states)[self.indices, actions]
        q_next = self.q_next(states_)
        q_eval = self.q_eval(states_)

        max_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[self.indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_cntr += 1
        self.decrement_eps()
