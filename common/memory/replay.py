import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, input_dims):
        self.capacity = capacity
        self.mem_cntr = 0
        self.state_mem = np.zeros((capacity, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros(capacity, dtype=np.int64)
        self.reward_mem = np.zeros(capacity, dtype=np.float32)
        self.new_state_mem = np.zeros((capacity, *input_dims),
                                      dtype=np.float32)
        self.done_mem = np.zeros(capacity, dtype=np.bool)

    def sample(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        dones = self.done_mem[batch]

        return states, actions, rewards, states_, dones

    def store(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.capacity
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.new_state_mem[idx] = state_
        self.done_mem[idx] = done

        self.mem_cntr += 1
