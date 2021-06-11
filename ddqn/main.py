import sys
sys.path.insert(0, '..')
from common.helpers.run import run
from common.helpers.env import Env
import torch
from agent import Agent
import gym

if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env = Env()
    input_dims = env.observation_space.shape
    batch_size = 16
    n_actions = env.action_space.n
    lr = 1e-4
    capacity = 15000
    episodes = 7500
    EPS = 1.0
    torch.cuda.empty_cache()
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    ddqn_agent = Agent(lr,
                       input_dims,
                       n_actions,
                       batch_size,
                       capacity,
                       env,
                       device,
                       EPS,
                       img_mode=False)

    # ddqn_agent.load()

    run(ddqn_agent, env, episodes)
