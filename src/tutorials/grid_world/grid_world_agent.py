from collections import defaultdict

import gymnasium as gym
import numpy as np


class GridWorldAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.01,
        initial_epsilon: float = 1,
        epsilon_decay: float = 0.002,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
    ):

        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def get_action(self, obs: tuple[int, int, int, int]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int, int],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
