from env.env import Env

if __name__ == '__main__':
    csv = "/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv"
    env = Env(csv)

    for i in range(2):
        print("Executing")
        done = False
        s = env.reset()
        while not done:
            a = 0
            s_, r, d, _ = env.step(a)
            env.display_grammian()
            s = s_
