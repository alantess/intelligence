import sys
sys.path.insert(0, '..')
from common.helpers.run import run
import gym

env = gym.make('CartPole-v0')
