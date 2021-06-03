import torch
import numpy as np
from memory.memory import ReplayMemory
from .network import DDQN


class Agent(object):
    def __init__(self,
                 lr,
                 n_actions,
                 candle_obs_shape,
                 gaf_obs_shape,
                 epsilon,
                 batch_size,
                 env,
                 capacity=100000,
                 eps_dec=4.5e-7,
                 fc1_dims=256,
                 fc2_dims=256,
                 replace=1000,
                 gamma=0.99):
        self.epsilon = epsilon
        self.scaler = torch.cuda.amp.GradScaler()
        self.eps_dec = eps_dec
        self.gamma = gamma
        self.env = env
        self.memory = ReplayMemory(capacity, candle_obs_shape, gaf_obs_shape)
        self.replace = replace
        self.eps_min = 0.01
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.update_cntr = 0

        # Evaluate Network
        self.q_eval = DDQN(lr, n_actions, 'eval', fc1_dims, fc2_dims)
        # Training Networking

        self.q_train = DDQN(lr, n_actions, 'trian', fc1_dims, fc2_dims)
        self.count_params()

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            candle_obs = torch.tensor([obs['candle']], dtype=torch.float).to(
                self.q_eval.device).permute(0, 3, 1, 2)
            gaf_obs = torch.tensor([obs['gaf']],
                                   dtype=torch.float).to(self.q_eval.device)

            # with torch.cuda.amp.autocast():
            actions = self.q_train.forward(candle_obs, gaf_obs)
            action = torch.argmax(actions).item()
        else:
            action = self.env.action_space.sample()

        return action

    def store_experience(self, s, a, r, s_, d):
        self.memory.store_transition(s, a, r, s_, d)

    def update_target_network(self):
        if self.update_cntr % self.replace == 0:
            self.q_eval.load_state_dict(self.q_train.state_dict())

    def save(self):
        print('Saving...')
        self.q_eval.save()
        self.q_train.save()

    def load(self):
        print('Loading...')
        self.q_eval.load()
        self.q_train.load()

    def count_params(self):
        model = self.q_train
        print('\nPARAMETERS: ',
              sum(p.numel() for p in model.parameters() if p.requires_grad))

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)

        candle_obs = torch.tensor(states['candle'],
                                  dtype=torch.float).to(self.q_eval.device)

        gaf_obs = torch.tensor(states['gaf'],
                               dtype=torch.float).to(self.q_eval.device)
        actions = torch.tensor(actions,
                               dtype=torch.int64).to(self.q_eval.device)
        rewards = torch.tensor(rewards,
                               dtype=torch.float).to(self.q_eval.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.q_eval.device)
        new_candle_obs = torch.tensor(states_['candle'],
                                      dtype=torch.float).to(self.q_eval.device)

        new_gaf_obs = torch.tensor(states_['gaf'],
                                   dtype=torch.float).to(self.q_eval.device)

        for p in self.q_train.parameters():
            p.grad = None

        self.update_target_network()

        q_pred = self.q_train.forward(candle_obs, gaf_obs)
        q_next = self.q_eval.forward(new_candle_obs, new_gaf_obs)
        q_eval = self.q_eval.forward(new_candle_obs, new_gaf_obs)

        max_actions = torch.argmax(q_eval, dim=1)
        q_next[dones] = 0.0
        y = rewards + self.gamma * q_next[max_actions]

        loss = self.q_train.loss(y, q_pred)
        loss.backward()

        self.q_train.optimizer.step()
        self.update_cntr += 1
        self.decrement_epsilon()
