import sys
sys.path.insert(0, '..')
from common.helpers.run import run
import torch
from agent import Agent
import gym

if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4')
    input_dims = env.observation_space.shape
    batch_size = 32
    n_actions = env.action_space.n
    lr = 1e-4
    capacity = 1000
    episodes = 7500
    EPS = 0.01
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    ddqn_agent = Agent(lr, input_dims, n_actions, batch_size, capacity, env,
                       device, EPS)

    # ddqn_agent.load()

    run(ddqn_agent, env, episodes)
