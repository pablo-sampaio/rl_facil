import gymnasium as gym
import time
import numpy as np

import torch

import dqn_models
import atari_wrappers

import collections


ENV_NAME = "PongNoFrameskip-v4"
ATARI_ENV = True
MODEL_FILE = "cap08/PongNoFrameskip-v4-2022-08-05,21-58-28-best.dat"

if ATARI_ENV:
    env = atari_wrappers.make_env_with_wrappers(ENV_NAME, render_mode="human")
else:
    env = gym.make(ENV_NAME, render_mode="human")

net = dqn_models.DQNNet(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(MODEL_FILE, map_location=lambda storage, loc: storage))

state, _ = env.reset()
total_reward = 0.0
c = collections.Counter()

while True:
    start_ts = time.time()
    state_v = torch.tensor(np.array([state], copy=False))

    q_vals = net(state_v)
    q_vals = q_vals.cpu().data.numpy()[0]
    action = np.argmax(q_vals)

    c[action] += 1
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if done:
        break

print("Total reward: %.2f" % total_reward)
print("Action counts:", c)
env.close()
