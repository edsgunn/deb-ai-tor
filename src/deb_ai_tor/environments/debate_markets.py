import json

from gymnasium import spaces
from pettingzoo import ParallelEnv


class Agent:
    pass


class DebateMarket(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "debate_market_v0"}

    def __init__(self, agents, initial_answer_price: float, render_mode: str):
        super().__init__(agents)
        self.initial_answer_price = initial_answer_price
        self.agents = agents

        # Observation and action spaces as strings
        self.observation_spaces = {
            agent: spaces.Text(max_length=1e6) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Text(max_length=1e6) for agent in self.agents
        }

        self.render_mode = render_mode
        self.step_count = 0

        self.holdings: dict[str, dict[Agent, float]] = None
        self.balances: dict[Agent, float] = None
        self.messsages: list[tuple[Agent, str]] = None

    def reset(self, question: str, answers=None, seed=None, options=None):
        self.step_count = 0
        self.holdings = {}
        self.balances = {agent: 1 for agent in self.agents}
        self.messages = []

        observations = [question for _ in self.agents]
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def get_initial_prompts(self):
        return ["beep" for agent in self.agents]

    def step(self, actions):
        self.step_count += 1
        answers = list(self.holdings.keys())
        new_answer_price = self.get_new_answer_price()
        answer_prices = self.get_answer_prices()
        message_price = self.get_message_price()

        extracted_actions = {
            agent: self.extract_actions(
                agent, action, answers, answer_prices, message_price
            )
            for agent, action in actions.items()
        }

        new_messages = []
        for agent, action in actions.items():
            balance = self.balances[agent]
            requests = json.loads(action)
            for request_type, info in requests.items():
                match request_type:
                    case "buy_message":
                        pass
                    case "buy_new_answer":
                        pass
                    case "buy_answer":
                        pass
                    case "sell_answer":
                        pass

    def get_new_answer_price(self):
        return 1

    def get_answer_prices(self):
        return [1 for _ in self.holdings.values()]

    def get_message_price(self):
        return 1

    def render(self):
        pass

    def close(self):
        pass

    def state(self):
        return {
            "holdings": self.holdings,
            "balances": self.balances,
            "messages": self.messages,
        }
