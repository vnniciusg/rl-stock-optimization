"""
A simple Q-Learning agent for discrete environments.

Author: vnniciusg
Email: vinnicius109@gmail.com
Date: 2025-08-31
License: MIT
"""

import random

import numpy as np


class QLearningAgent:
    """Q-learning agent for discrete state and action spaces."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update Q-table using the Q-Learning formula."""
        old_q = self.q_table[state, action]
        future_q = np.max(self.q_table[next_state])
        self.q_table[state, action] = old_q + self.learning_rate * (
            reward + self.discount_factor * future_q - old_q
        )
