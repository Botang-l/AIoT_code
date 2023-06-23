import gymnasium as gym
import torch
import torch.optim as optim

from Memory import *
from Model import *
from Environment import *
import configparser
import sys


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#env = gym.make("CartPole-v1")
env = Environment()

config = configparser.ConfigParser()
config.read('config.ini')

args = sys.argv
arg = args[-1] if(len(args) == 2) else 'DEFAULT'

try:
    print('使用', arg, '作為超參數')
    BATCH_SIZE = int(config[arg]['BATCH_SIZE'])
    GAMMA = float(config[arg]['GAMMA'])
    EPS_START = float(config[arg]['EPS_START'])
    EPS_END = float(config[arg]['EPS_END'])
    EPS_DECAY = int(config[arg]['EPS_DECAY'])
    TAU = float(config[arg]['TAU'])
    LR = float(config[arg]['LR'])
    GPU_TIMES = int(config[arg]['GPU_TIMES'])
    CPU_TIMES = int(config[arg]['CPU_TIMES'])
    MEM_SIZE = int(config[arg]['MEM_SIZE'])

except:
    print('參數輸入錯誤')
    sys.exit()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()

n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEM_SIZE)

steps_done = 0

episode_durations = []
total_reward = []
