from traffic_env import TrafficIntersectionEnv

class MultiIntersectionEnv:
    def __init__(self, num_intersections):
        self.num_intersections = num_intersections
        self.intersection_ids = list(range(num_intersections))

        # create one env per intersection
        self.envs = {
            i: TrafficIntersectionEnv()
            for i in self.intersection_ids
        }

        # shared spaces (same for all)
        self.action_size = self.envs[0].action_space.n
        self.state_size = self.envs[0].observation_space.shape[0]

    def reset(self):
        states = {}
        for i, env in self.envs.items():
            obs, _ = env.reset()
            states[i] = self._state_to_tuple(obs)
        return states

    def step(self, actions):
        next_states = {}
        rewards = {}
        done = False
        info = {}

        for i, action in actions.items():
            obs, reward, terminated, truncated, _ = self.envs[i].step(action)

            next_states[i] = self._state_to_tuple(obs)
            rewards[i] = reward
            done = done or terminated or truncated
            info[i] = {}

        return next_states, rewards, done, info

    def _state_to_tuple(self, obs):
        # IMPORTANT: Q-learning needs hashable states
        return tuple(obs.tolist())
