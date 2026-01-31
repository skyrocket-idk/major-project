class MultiAgentTrainer:
    def __init__(self, env, agents, neighbors=None, beta=0.3):
        self.env = env
        self.agents = agents
        self.neighbors = neighbors or {}
        self.beta = beta

    def shape_rewards(self, rewards):
        shaped = {}
        for i in rewards:
            neighbor_reward = sum(
                rewards.get(j, 0) for j in self.neighbors.get(i, [])
            )
            shaped[i] = rewards[i] + self.beta * neighbor_reward
        return shaped

    def train_episode(self):
        states = self.env.reset()
        done = False
        total_reward = 0

        while not done:

            # 1️⃣ Select actions using current policy
            actions = {
                i: agent.select_action(states[i])
                for i, agent in self.agents.items()
            }

            # 2️⃣ Step environment (this increments timestep)
            next_states, rewards, done, info = self.env.step(actions)

            # 3️⃣ Optional cooperative reward shaping
            rewards = self.shape_rewards(rewards)

            # 4️⃣ Q-learning updates
            for i, agent in self.agents.items():
                agent.update(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i]
                )
                total_reward += rewards[i]

            # 5️⃣ Advance state
            states = next_states

        # episode length (all envs are synchronous)
        episode_length = self.env.envs[0].timestep

        # normalized metric (this is what you should log)
        avg_reward_per_step = total_reward / episode_length

        # epsilon decay ONCE per episode
        for agent in self.agents.values():
            agent.decay_epsilon()

        return avg_reward_per_step

