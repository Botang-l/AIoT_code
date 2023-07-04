import torch
import matplotlib
import matplotlib.pyplot as plt
from . import *
from itertools import count
import os

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)    # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            plt.show()
            display.display(plt.gcf())


def RL_model():
    policy_net = torch.load(os.path.join(os.path.dirname(__file__), 'model/policy_net.pth'))
    target_net = torch.load(os.path.join(os.path.dirname(__file__), 'model/target_net.pth'))

    print('最終決策 policy net:')
    # Initialize the environment and get it's state
    state, _ = env.reset()
    #env.displayPosition()
    print('-' * 10)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = policy_net(state).max(1)[1].view(1, 1)
        #print(policy_net(state))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            episode_durations.append(t + 1)
            total_reward.append(env.totalReward())
            #plot_durations(total_reward)
            break
    env.displayTotalReward()
    print(env.actionlist)

    print('最終決策 target net:')
    # Initialize the environment and get it's state
    state, _ = env.reset()
    #env.displayPosition()
    print('-' * 10)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        #print(state)

        action = target_net(state).max(1)[1].view(1, 1)
        #print(target_net(state))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        if done:
            episode_durations.append(t + 1)
            total_reward.append(env.totalReward())
            #plot_durations(total_reward)
            break
    env.displayTotalReward()
    print(env.actionlist)
