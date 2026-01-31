import numpy as np
import pickle
from gymnasium import Env
from gymnasium.spaces import Discrete, Box


#                TRAFFIC ENVIRONMENT

class TrafficIntersectionEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Actions:
        # 0 = NS green
        # 1 = EW green
        # 2 = All red (transition)
        # 3 = Pedestrian mode (optional)
        self.action_space = Discrete(2)

        # Observations:
        # [queue_NS, queue_EW, phase]
        self.observation_space = Box(
            low=np.array([0,0,0]),
            high=np.array([50,50,3]),
            dtype=np.int32
        )

        # internal states
        self.queue_NS = 0
        self.queue_EW = 0
        self.phase = 0      # current signal phase (0–3)
        self.timestep = 0

        # stats
        self.total_waiting_time = 0

        # config
        self.arrival_rate_NS = 0.25
        self.arrival_rate_EW = 0.25
        self.pass_rate = 1
        self.max_steps = 200



    def reset(self, seed=None, options=None):
        self.prev_phase = 0
        super().reset(seed=seed)
        self.queue_NS = 0
        self.queue_EW = 0
        self.phase = 0
        self.timestep = 0
        self.total_waiting_time = 0
        return self._get_obs(), {}



    def step(self, action):
        self.timestep += 1
        self.phase = action
        self.queue_NS = min(self.queue_NS, 50)
        self.queue_EW = min(self.queue_EW, 50)
        self.prev_phase = 0

        # spawn new vehicles
        if np.random.rand() < self.arrival_rate_NS:
            self.queue_NS += 1
        if np.random.rand() < self.arrival_rate_EW:
            self.queue_EW += 1

        # cars pass if green
        if action == 0:    # NS green
            self.queue_NS = max(0, self.queue_NS - self.pass_rate)
        elif action == 1:  # EW green
            self.queue_EW = max(0, self.queue_EW - self.pass_rate)
        if action == 0:
            self.queue_NS = max(0, self.queue_NS - self.pass_rate)
        elif action == 1:
            self.queue_EW = max(0, self.queue_EW - self.pass_rate)

        # waiting time accumulates
        self.total_waiting_time += (self.queue_NS + self.queue_EW)
        switch_penalty = 0
        if action != self.prev_phase:
            switch_penalty = -0.5
        self.prev_phase = action


        # reward = - (total queue + waiting)
        reward = - (self.queue_NS + self.queue_EW) + switch_penalty


        terminated = False
        truncated = self.timestep >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}


    # --------------------------------------------------------

    def _get_obs(self):
        return np.array([self.queue_NS, self.queue_EW, self.phase], dtype=np.int32)

