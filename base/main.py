from env.env import Env

if __name__ == '__main__':
    csv = "/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv"
    env = Env(csv)
    s = env.reset()

    # env.display_grammian()
    # env.show()
    # print(s.shape)
