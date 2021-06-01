from env.env import Env

if __name__ == '__main__':
    csv = "/media/alan/seagate/Downloads/BTCUSDT_Binance_futures_data_hour (1).csv"
    env = Env(csv)
    env.plot_animation()
