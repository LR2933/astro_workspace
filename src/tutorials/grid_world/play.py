import argparse
from collections import defaultdict

import gymnasium as gym
import numpy as np

from grid_world_agent import GridWorldAgent
from utils import encode_obs, load_model

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")
agent = GridWorldAgent(env=env)

parse = argparse.ArgumentParser()
parse.add_argument("--model")

model_path = parse.parse_args().model

q_values = load_model(model_path)
q_values = defaultdict(
    lambda: np.zeros(env.action_space.n),
    q_values,
)

while True:
    obs, info = env.reset()
    done = False

    while not done:
        action = np.argmax(q_values[encode_obs(obs)])

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(encode_obs(obs), action, reward, terminated, encode_obs(next_obs))

        done = terminated or truncated
        obs = next_obs
