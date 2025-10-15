import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class DebateSandwichEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "debate_sandwich_v0"}

    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }
        self.state = None

    def reset(self, seed=None, options=None):
        self.state = {agent: np.zeros(4, dtype=np.float32) for agent in self.agents}
        self.num_rounds = options.get("num_rounds", 2)
        observations = self.state.copy()
        return observations

    def step(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {
            agent: np.random.rand(4).astype(np.float32) for agent in self.agents
        }
        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        print("Current state:", self.state)

    def close(self):
        pass
