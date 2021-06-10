from gym import spaces
import numpy as np
import pandas as pd
from sklearn import preprocessing
import cudf


class Env(object):
    """
    A reinforcment learning environment made for cryptocurrency. 
    Ability to trade, sell, and hold crypto
    Example:
    for i in range(10):
        s = env.reset()
        done = False
        while not done:
            a = env.actions_space.sample()
            s_, reward, done, _ = env.step(a)
            s = s_
    """
    def __init__(
            self,
            csv="/media/alan/seagate/Downloads/Binance_LTCUSDT_minute_ds.csv",
            n_actions=3,
            stop_loss=0.8,
            time_inc=1):
        self.csv = csv
        self.cols = 0
        self.observation_space = None
        self.time_inc = time_inc
        self.stop_loss = stop_loss
        self.action_set = np.arange(n_actions)
        self.actions_space = spaces.Discrete(len(self.action_set))
        self.unscaled_df = None
        self.scaler = preprocessing.StandardScaler()
        self.df = None
        self.price = None
        self.usdt_wallet = None
        self.crypto_wallet = None
        self.n_steps = None
        self.reward_dec = 1.0
        self.reward_sub = 0.99e-3
        self.cur_step = 0
        self.starting_amount = 0
        self._load()

    # Loads observations space / Sends dataframe to GPU
    def _load(self):
        df = pd.read_csv(self.csv)
        df = df.iloc[::-1]
        df = df.reset_index()
        df = df.drop(
            ['date', 'index', 'unix', 'symbol', 'Volume USDT', 'tradecount::'],
            axis=1)
        self.n_steps, self.cols = df.shape
        self.observation_space = np.zeros((self.cols + 2), dtype=np.float32)
        scaled_df = self.scaler.fit_transform(df)
        self.unscaled_df = cudf.DataFrame(df)
        self.df = cudf.DataFrame(scaled_df)
        self.df.columns = ['open', 'high', 'low', 'close', 'Volume LTC']

    def reset(self):
        self.cur_step = 0
        self.crypto_wallet = 0
        self.usdt_wallet = np.random.randint(300, 3000)
        self._get_price()
        self.starting_amount = self.usdt_wallet
        self.reward_dec = self.reward_dec - self.reward_sub if self.reward_dec > 0 else 0
        return self._get_obs()

    def step(self, a):
        assert (0 <= a <= len(self.action_set)), "Invalid Action"
        reward = 0
        prev_holdings = self._get_holdings()
        self._trade(a)
        self.cur_step += self.time_inc
        self._update()
        cur_holdings = self._get_holdings()
        profit = cur_holdings / prev_holdings

        if cur_holdings < self.stop_loss * self.starting_amount:
            done = True
        else:
            done = (self.time_inc <= self.n_steps - self.time_inc)

        info = {'btc': self.crypto_wallet, 'usdt': self.usdt_wallet}

        if profit > 1.0:
            reward += (profit * self.reward_dec) + 1
        else:
            reward += (profit * self.reward_dec) - 1

        return self._get_obs(), reward, done, info

    def _update(self):
        self.crypto_wallet *= float(
            self.unscaled_df.values[self.cur_step + self.time_inc][3] /
            self.price)

    def _trade(self, a):
        if a == 0:
            return
        if a == 1:
            self._buy_or_sell(True)

        if a == 2:
            self._buy_or_sell(False)

    def _buy_or_sell(self, purchase):
        if purchase:
            if self.usdt_wallet >= self.price:
                self.usdt_wallet -= self.price
                self.crypto_wallet += self.price
            else:
                return
        else:
            if self.crypto_wallet >= self.price:
                self.crypto_wallet -= self.price
                self.usdt_wallet += self.price
            else:
                return

    def _get_holdings(self):
        return self.crypto_wallet + self.usdt_wallet

    def _get_obs(self):
        state = self.observation_space
        state[:self.cols] = self.df.values[self.cur_step].get()
        state[self.cols] = self.crypto_wallet
        state[self.cols + 1] = self.usdt_wallet
        return state

    def _get_price(self):
        self.price = self.unscaled_df.values[self.cur_step][3]
