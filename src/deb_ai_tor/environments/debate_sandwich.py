import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class DebateSandwichEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "debate_sandwich_v0"}

    def __init__(self, agents):
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.observation_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }
        self.initial_answers = None
        self.final_anwers = None

    def reset(self, seed=None, options=None):
        self.state = {agent: np.zeros(4, dtype=np.float32) for agent in self.agents}
        self.num_rounds = options.get("num_rounds", 2)

        required_keys = ["question", "answers"]
        for key in required_keys:
            if key not in options:
                raise ValueError(f"Missing required option: {key}")
        self.question = options["question"]
        self.answers = {chr(65 + i): ans for i, ans in enumerate(options["answers"])}

        self.current_round = 0
        observation_message = "Which of the following answers do you think is best?\n"
        observation_info = f"Question: {self.question}\nAnswers: {self.answers}\n"
        observation = {
            "message": observation_message,
            "info": observation_info,
        }
        observations = {agent: observation for agent in self.agents}
        return observations

    def step(self, actions):
        self.current_round += 1

        if self.current_round < self.num_rounds:
            if self.current_round == 1:
                self.initial_answers = actions
            if self.current_round == self.num_rounds - 1:
                observation_message = (
                    "Which of the following answers do you now think is best?\n"
                )
            else:
                observation_message = "Debate your choice of answer\n"
            observation = {"message": observation_message, "info": actions}
            observations = {agent: observation for agent in self.agents}
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: False for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
        elif self.current_round == self.num_rounds:
            self.final_answers = actions
            observations = {agent: None for agent in self.agents}
            rewards = self.get_final_rewards()
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
        else:
            raise ValueError("Environment has already terminated.")

        return observations, rewards, terminations, truncations, infos

    def get_final_rewards(self):
        switches_by_answer = {}
        for key in self.answers.keys():
            initial = sum(key == val for val in self.initial_answers.values())
            final = sum(key == val for val in self.final_answers.values())
            switches_by_answer[key] = final - initial

        return {
            agent: switches_by_answer.get(self.initial_answers[agent], 0)
            for agent in self.agents
        }

    def render(self, mode="human"):
        print("Current state:", self.state)

    def close(self):
        pass
