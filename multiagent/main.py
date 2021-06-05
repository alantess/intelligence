import sys
sys.path.insert(0, '..')
import gym

# from common.env.env import Env
from time import time
import torch
import torch.multiprocessing as mp
import numpy as np
from agent import Agent
from network import *

env = gym.make('SpaceInvaders-v0')


def run(agent, track_score):
    cur_step = 0
    EPISODES = 2000
    for epi in range(EPISODES):
        start_time = time()
        score = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, state_, done)
            score += reward
            if cur_step % 4 == 0:
                agent.learn()

            # if (cur_step + 1) % 1000 == 0 or done:
            #     agent.update_global_network()

            state = state_
            cur_step += 1
        print(
            f"Episode({epi}) {agent.name}: SCORE: {score:.2f} Epsilon: {agent.epsilon:.4f} Time: {time() - start_time:.3f}"
        )
        if score > track_score.value:
            track_score.value = int(score)
            agent.update_global_network()
            print(f"{agent.name} - BEST SCORE: {track_score.value}")


if __name__ == '__main__':
    channels = env.observation_space.shape[2]
    n_actions = env.action_space.n
    lr = 1e-4
    batch_size = 16
    eps = 1.0
    eps_dec = 3.5e-6
    capacity = 5000
    device = torch.device('cuda')

    model = MULTIDQN(lr=lr, input_dims=channels, n_actions=n_actions)
    model.share_memory()
    optimizer = SharedAdam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mp.set_start_method('spawn')
    score = mp.Value('i', 0)
    start = time()

    workers = [
        Agent(model, optimizer, lr, channels, n_actions, batch_size, capacity,
              eps, eps_dec, env, device, i) for i in range(3)
    ]
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    processes = [mp.Process(target=run, args=(w, score)) for w in workers]
    [p.start() for p in processes]
    [p.join() for p in processes]

    # Pool Method
    # with mp.Pool(processes=len(workers)) as pool:
    #     x = [pool.map(run, workers)]
    print(f'Total time: {time() - start}')