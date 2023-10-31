import gymnasium as gym
import time
import numpy as np

import torch

import dqn_models
import atari_wrappers

import collections


ENV_NAME = "PongNoFrameskip-v4"
ATARI_ENV = True
MODEL_FILE = "cap08novo/PongNoFrameskip-v4-agente-treinado.net"

VIDEO_DIR = None 
RENDER = True

DESIRED_FPS = 25


if ATARI_ENV:
    env = atari_wrappers.make_env_with_wrappers(ENV_NAME)
else:
    env = gym.make(ENV_NAME)

if VIDEO_DIR is not None:
    env = gym.wrappers.Monitor(env, VIDEO_DIR)

net = dqn_models.DQNNet(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load(MODEL_FILE, map_location=lambda storage, loc: storage))

state = env.reset()
if RENDER:
    env.render()

total_reward = 0.0
c = collections.Counter()

while True:
    start_ts = time.time()
    state_v = torch.tensor(np.array([state], copy=False))
    q_vals = net(state_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if RENDER:
        env.render()
        delta = 1/DESIRED_FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    if done:
        break

print("Total reward: %.2f" % total_reward)
print("Action counts:", c)
env.close()
