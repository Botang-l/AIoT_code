import gymnasium as gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import *
from .Memory import *
from .Model import *
from .Strategy import *
from .util import *
from .Environment import *


def dqn():
    num_episodes = GPU_TIMES if torch.cuda.is_available() else CPU_TIMES
    for i in range(1):
        for i_episode in range(num_episodes):
            print('第{}輪決策:'.format(i_episode + 1))
            # Initialize the environment and get it's state
            state, _ = env.reset()
            #env.displayPosition()
            print('-' * 10)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                if (t > 99):
                    break
                if (t > 90):
                    #     state[0][1] = t
                    print(state)
                action = select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key
                                         ] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    total_reward.append(env.TotalReward())
                    #plot_durations(total_reward)
                    break
            env.displayTotalReward()

        torch.save(target_net, os.path.join(os.path.dirname(__file__), 'model/target_net.pth'))
        torch.save(policy_net, os.path.join(os.path.dirname(__file__), 'model/policy_net.pth'))

        print('Complete')
        plot_durations(total_reward, show_result=True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'result/reward{}.png'.format(str(i + 1))))
        plot_durations(episode_durations, show_result=True)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'result/episode_durations{}.png'.format(str(i + 1))))

        plt.ioff()
        plt.show()
