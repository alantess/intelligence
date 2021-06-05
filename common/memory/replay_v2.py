import torch
import numpy as np
from torchvision import transforms


class ReplayBuffer:
    r"""
    Capacity: Max size of the replay buffer
    Obs_dims: Preferred image size of the observation
    """
    def __init__(self, capacity, obs_dims):
        self.max_mem = capacity
        self.mem_cntr = 0
        self.image_dims = (3, obs_dims, obs_dims)
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((obs_dims, obs_dims))])

        self.state_memory = torch.zeros((capacity, *self.image_dims),
                                        dtype=torch.float)
        self.action_memory = torch.zeros(capacity, dtype=torch.int64)
        self.reward_memory = torch.zeros(capacity, dtype=torch.float)

        self.new_state_memory = torch.zeros((capacity, *self.image_dims),
                                            dtype=torch.float)
        self.terminal_memory = torch.zeros(capacity, dtype=torch.bool)

    def _experience_to_tensor(self, state, action, reward, state_, done):
        state = self.transforms(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        state_ = self.transforms(state_)
        done = torch.tensor(done)

        return state, action, reward, state_, done

    def store_experience(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.max_mem
        state, action, reward, state_, done = self._experience_to_tensor(
            state, action, reward, state_, done)

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.max_mem)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        s = self.state_memory[batch]
        a = self.action_memory[batch]
        r = self.reward_memory[batch]
        s_ = self.new_state_memory[batch]
        d = self.terminal_memory[batch]
        return s, a, r, s_, d
