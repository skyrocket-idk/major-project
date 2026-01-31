from collections import defaultdict
import numpy as np
import random

class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        agent_id=None,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    ):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self, min_eps=0.05, decay=0.995):
        self.epsilon = max(min_eps, self.epsilon * decay)
