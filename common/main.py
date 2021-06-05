import time
import numpy as np
from env.env import Env
from agent.agent import Agent

if __name__ == '__main__':
    csv = "/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv"
    lr = 1e-4
    epsilon = 1
    EPISODE = 1000
    BATCH = 16
    EPS_DEC = 8.5e-4
    capacity = 16000
    best_score = -np.inf
    n_steps = 0
    env = Env(csv, 60)

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
                agent.learn()
            s = s_
            n_steps += 1
        if score > best_score:
            best_score = score
            agent.save()
        print(f"Episode {i} - Score: {score} - Best: {best_score:.2f} \n \
             Epsilon {agent.epsilon:.6f}\n \
             Reward Dec: {env.reward_dec:.2f} - Info {info}")

        print("--- %s seconds ---" % (time.time() - start))
