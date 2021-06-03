import time
import numpy as np
from env.env import Env
from agent.agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    csv = "/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv"
    lr = 1e-4
    epsilon = 1
    EPISODE = 1000
    BATCH = 32
    EPS_DEC = 9e-5
    capacity = 15000
    best_score = -np.inf
    running_avg, scores = [], []
    n_steps = 0
    env = Env(csv, 60)
    s = env.reset()

    print("Executing")
    agent = Agent(lr, env.action_set.shape[0], env.candle_obs_space.shape,
                  env.grammian_obs_space.shape, epsilon, BATCH, env, capacity,
                  EPS_DEC)

    for i in range(EPISODE):
        score = 0
        start = time.time()
        done = False

        s = env.reset()
        while not done:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            score += r

            agent.store_experience(s, a, r, s_, done)
            if n_steps % 4 == 0:
                print(info)
                print(f'epsilon: {agent.epsilon:.6f}')
                agent.learn()
            s = s_
            n_steps += 1
        scores.append(score)
        avg_score = np.mean(scores[-50:])
        running_avg.append(avg_score)
        if avg_score > best_score:
            plt.plot(running_avg)
            plt.savefig('scores.png')
            best_score = avg_score
            agent.save()
        print(
            f"Episode {i} - Score: {score} - Best: {best_score:.2f} - Epsilon {agent.epsilon:.6f}\n Reward Dec: {env.reward_dec:.2f} - Info {info} - AVG: {avg_score:.3f}"
        )

        print("--- %s seconds ---" % (time.time() - start))
