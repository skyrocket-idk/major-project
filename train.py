from multi_intersection_env import MultiIntersectionEnv
from agent_factory import build_agents
from marl_trainer import MultiAgentTrainer
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from random_policy import run_random_policy
from random_policy import run_fixed_time_policy
# create 2 intersections to start
env = MultiIntersectionEnv(num_intersections=2)

agents = build_agents(env)

trainer = MultiAgentTrainer(env, agents)


episode_rewards = []

for episode in range(1000):
    avg_reward = trainer.train_episode()
    episode_rewards.append(avg_reward)

    if episode % 50 == 0:
        print(f"Episode {episode} | Avg Reward/Step: {avg_reward:.3f}")

with open("episode_rewards.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "avg_reward_per_step"])
    for i, r in enumerate(episode_rewards):
        writer.writerow([i, r])

random_baseline = run_random_policy(env, num_episodes=100)
print("Random Policy Avg Reward/Step:", random_baseline)

fixed_baseline = run_fixed_time_policy(env)
print("Fixed-Time Policy Avg Reward/Step:", fixed_baseline)

os.makedirs("plots", exist_ok=True)

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode="valid")

# Raw rewards
plt.figure()
plt.plot(episode_rewards, alpha=0.3)
plt.xlabel("Episode")
plt.ylabel("Avg Reward per Step")
plt.title("MARL Training Performance (Raw)")
plt.savefig("plots/reward_raw.png", dpi=300, bbox_inches="tight")
plt.close()

# Smoothed rewards
ma_rewards = moving_average(episode_rewards, window=50)

plt.figure()
plt.plot(ma_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg Reward per Step (Moving Avg)")
plt.title("MARL Training Performance (Smoothed)")
plt.savefig("plots/reward_moving_avg.png", dpi=300, bbox_inches="tight")
plt.close()

