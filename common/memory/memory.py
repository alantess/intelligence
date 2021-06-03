import numpy as np


class ReplayMemory():
    def __init__(self, capacity, candle_obs_space, grammian_obs_space):
        self.max_mem = capacity
        self.mem_cntr = 0
        self.state_memory = {
            'candle': np.zeros((capacity, *candle_obs_space),
                               dtype=np.float32),
            'gaf': np.zeros((capacity, *grammian_obs_space), dtype=np.float32)
        }
        self.action_memory = np.zeros((capacity), dtype=np.int64)
        self.reward_memory = np.zeros(capacity, dtype=np.float32)

        self.new_state_memory = {
            'candle': np.zeros((capacity, *candle_obs_space),
                               dtype=np.float32),
            'gaf': np.zeros((capacity, *grammian_obs_space), dtype=np.float32)
        }
        self.terminal_memory = np.zeros(capacity, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.max_mem
        self.state_memory['candle'][idx] = state['candle']
        self.state_memory['gaf'][idx] = state['gaf']
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory['candle'][idx] = state_['candle']
        self.new_state_memory['gaf'][idx] = state_['gaf']
        self.terminal_memory[idx] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size=3):
        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = {
            'candle': self.state_memory['candle'][batch],
            'gaf': self.state_memory['gaf'][batch]
        }
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = {
            'candle': self.new_state_memory['candle'][batch],
            'gaf': self.new_state_memory['gaf'][batch]
        }
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
