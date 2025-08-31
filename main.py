import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env import InventoryEnv
from src.q_learning import QLearningAgent

env = InventoryEnv()

agent = QLearningAgent(
    state_size=env.observation_space.n,
    action_size=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1,
)

episodes = 5000
rewards_history = []

for episode in tqdm(range(episodes), desc="Training Episodes"):
    state = env.reset()
    total_reward = 0
    done = False

    for step in range(50):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    rewards_history.append(total_reward)


plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title("Q-Learning Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("assets/training_rewards.png")
plt.close()

policy = [agent.choose_action(s) for s in range(env.observation_space.n)]
plt.figure(figsize=(10, 5))
plt.bar(range(env.observation_space.n), policy)
plt.title("Learned Policy (Action per Stock Level)")
plt.xlabel("Stock Level (State)")
plt.ylabel("Action (Order Quantity)")
plt.grid(True)
plt.tight_layout()
plt.savefig("assets/learned_policy.png")
plt.close()
