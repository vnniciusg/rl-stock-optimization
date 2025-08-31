"""
A simple reinforcement learning environment for inventory management.

Author: vnniciusg
Email: vinnicius109@gmail.com
Date: 2025-08-31
License: MIT
"""

import gymnasium as gym
from gymnasium import spaces


class InventoryEnv(gym.Env):
    """
    Inventory Management Envonment

    This environment simulates a simple inventory management scenario where an agent must decide
    how many items to order to keep the stock within acceptable limites while minimizing penalties.

    State:
        - An integer representating  the current stock level (0 to max_stock).

    Action Space:
        - 0: Order 0 items
        - 1: Order 1 items
        - 2: Order 2 items
        - 3: Order 3 items

    Observation Space:
        - 0 to 10: Current inventory level

    Reward:
        - -10 if stock is 0 (penalty for stockout)
        - -5 if stock is ate max_stock (penalty for overstock)
        - 0 otherwise (ideal stock level)
    """

    def __init__(self) -> None:
        super(InventoryEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(11)
        self.state: int = 5
        self.max_stock: int = 10
        self.min_stock: int = 0
        self.demand: int = 3

    def reset(self) -> int:
        """Resets the environment to the initial state."""
        self.state = 5
        return self.state

    def step(self, action: int) -> tuple[int, int, bool, dict]:
        """
        Executes one step in the environment.

        Args:
            action (int): Number of items to order (0, 1, 2, or 3)

        Returns:
            tuple:
                state (int): New stock level
                reward (int): Reward based on stock level
                done (bool): Whether the episode is finished (always False here)
                info (dict): Additional info (empty dict)
        """
        self.state = min(
            self.max_stock, max(self.min_stock, self.state + action - self.demand)
        )

        if self.state == 0:
            reward = -10
        elif self.state == self.max_stock:
            reward = -5
        else:
            reward = 0

        done = False
        return self.state, reward, done, {}

    def render(self) -> None:
        """Prints the current stock level."""
        print(f"Current stock: {self.state}")
