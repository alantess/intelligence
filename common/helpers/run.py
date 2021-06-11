import numpy as np
from time import time


def run(agent, env, n_epsidoes):
    np.random.seed(1337)
    scores = []
    cur_step = 0
    best = -np.inf
    print('Starting...')
    for epi in range(n_epsidoes):
        start_time = time()
        done = False
        state = env.reset()
        score = 0
        while not done:
            # env.render()
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            agent.store_experience(state, action, reward, state_, done)
            agent.learn()
            state = state_
            cur_step += 1
        scores.append(score)
        agent_score = float(score)
        # print(f"Episode {epi}")
        # print(f"{score:.2f}")
        # print(f"Best {best:.2f}")
        # print(f"eps: {agent.epsilon:.6f}")
        # print(f"Step: {cur_step}")
        # print(f"{time() - start_time:.2f} secs")
        print(
            f"Episode({epi}): SCORE: {agent_score:.2f} BEST: {best:.2f} EPS: {agent.epsilon:.6f}  Steps: {cur_step} \nTime: {time() - start_time:.2f}"
        )
        if epi % 10 == 0:
            avg = np.mean(scores)
            scores = []
            if avg > best:
                best = avg
                agent.save()

    # env.close()
