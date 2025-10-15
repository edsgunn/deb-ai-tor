import json
import math
import random
import string
import uuid
from collections import defaultdict, deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class Agent:
    pass


class DebateMarket(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "debate_market_v0"}

    def __init__(
        self,
        agents: list[str],
        liquidity_b: float = 10.0,
        msg_window: int = 10,
        render_mode: str = "human",
    ):
        self.agents = agents
        self.render_mode = render_mode
        self.b = liquidity_b
        self.msg_window = msg_window

        self.observation_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Text(max_length=1_000_000) for agent in self.agents
        }

        # State
        self.q: np.ndarray | None = None
        self.answers: list[str] = []
        self.answer_ids: list[str] = []  # immutable ids for answers
        self.answer_id_to_idx: dict[str, int] = {}
        self.holdings: dict[str, np.ndarray] = {}
        self.balances: dict[str, float] = {}
        self.messages_all: list[tuple[str, str]] = []  # full history
        self.messages_round: list[tuple[str, str]] = []  # current round

        self.msg_counts = deque(maxlen=self.msg_window)
        self.step_count = 0

    # ------------------------------
    # LMSR functions
    # ------------------------------
    def cost(self, q: np.ndarray) -> float:
        # Numerically stable log-sum-exp
        scaled_q = q / self.b
        m = np.max(scaled_q)
        return self.b * (m + math.log(np.exp(scaled_q - m).sum()))

    def prices(self, q: np.ndarray) -> np.ndarray:
        exp_q = np.exp(q / self.b)
        return exp_q / exp_q.sum()

    def trade_cost(self, dq: np.ndarray) -> float:
        return self.cost(self.q + dq) - self.cost(self.q)

    # ------------------------------
    # Message cost (recent only)
    # ------------------------------
    def message_price(self) -> float:
        n_recent = sum(self.msg_counts) if self.msg_counts else 0
        return 0.05 * (1 + math.log1p(n_recent))

    # ------------------------------
    # PettingZoo API
    # ------------------------------
    def reset(self, question: str, answers=None, seed=None, options=None):
        self.step_count = 0
        self.answers = answers or []
        self.answer_ids = []
        self.answer_id_to_idx = {}
        self.q = np.zeros(len(self.answers), dtype=float)
        # fixed_answers is True if answers are provided, False otherwise
        self.fixed_answers = answers is not None
        self.round_started_with_zero_answers = len(self.answers) == 0
        # Generate answer IDs for all initial answers
        for i, ans in enumerate(self.answers):
            new_id = self._generate_short_id()
            while new_id in self.answer_id_to_idx:
                new_id = self._generate_short_id()
            self.answer_ids.append(new_id)
            self.answer_id_to_idx[new_id] = i
        self.holdings = {
            agent: {aid: 0.0 for aid in self.answer_ids} for agent in self.agents
        }
        self.balances = {agent: 1.0 for agent in self.agents}
        self.messages_all = []
        self.messages_round = []
        self.msg_counts.clear()
        observations = {
            agent: {
                "balance": self.balances[agent],
                "holdings": {aid: self.holdings[agent][aid] for aid in self.answer_ids},
                "prices": [],
                "answers": [
                    {"id": aid, "text": self.answers[self.answer_id_to_idx[aid]]}
                    for aid in self.answer_ids
                ],
                "message_price": self.message_price(),
                "messages": [],
            }
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _generate_short_id(self, length=5):
        # Generates a short, readable, human-friendly id
        chars = string.ascii_uppercase + string.digits
        return "".join(random.choices(chars, k=length))

    def _add_new_answer(self, ans_text, dq_by_agent=None):
        self.answers.append(ans_text)
        new_id = self._generate_short_id()
        while new_id in self.answer_id_to_idx:
            new_id = self._generate_short_id()
        self.answer_ids.append(new_id)
        self.answer_id_to_idx[new_id] = len(self.answers) - 1
        self.q = np.append(self.q, 0.0)
        for ag in self.agents:
            self.holdings[ag][new_id] = 0.0
            if dq_by_agent is not None:
                dq_by_agent[ag] = np.append(dq_by_agent[ag], 0.0)
        return new_id

    def _max_shares_for_amount(self, idx, amount, market_was_empty):
        # If the market was empty at the start of the step, agent is creating a new answer: equity = amount paid
        if market_was_empty:
            return amount
        # Otherwise, use AMM pricing
        left, right = 0.0, 1e6  # reasonable upper bound
        for _ in range(20):
            mid = (left + right) / 2
            dq = np.zeros_like(self.q)
            dq[idx] = mid
            cost = self.trade_cost(dq)
            if cost > amount:
                right = mid
            else:
                left = mid
        # If left is 0, check if a minimal share can be bought (for non-empty market)
        if left == 0.0:
            dq = np.zeros_like(self.q)
            dq[idx] = 1e-8
            cost = self.trade_cost(dq)
            min_cost = 0.01 * len(self.answers)
            if cost <= amount and cost >= min_cost:
                return 1e-8
        return left

    def _remove_answer(self, idx):
        # Remove answer at idx from all relevant structures
        del self.answers[idx]
        aid = self.answer_ids[idx]
        del self.answer_ids[idx]
        del self.answer_id_to_idx[aid]
        self.q = np.delete(self.q, idx)
        for ag in self.agents:
            del self.holdings[ag][aid]
        # Rebuild id->idx mapping
        self.answer_id_to_idx = {aid: i for i, aid in enumerate(self.answer_ids)}

    def step(self, actions):
        self.step_count += 1
        num_answers_at_start = len(self.answers)
        min_investment = (
            0.01 * num_answers_at_start if num_answers_at_start > 0 else 0.0
        )
        rewards = {agent: 0.0 for agent in self.agents}
        self.messages_round = []
        dq_by_agent = {agent: np.zeros_like(self.q) for agent in self.agents}
        new_answers = {}
        new_msgs = []
        buy_requests = []

        for agent, action_str in actions.items():
            if not action_str:
                continue
            try:
                requests = json.loads(action_str)
            except json.JSONDecodeError:
                continue

            for request in requests:
                match request["type"]:
                    case "buy_answer":
                        buy_requests.append(
                            (agent, request["answer_id"], float(request["amount"]))
                        )
                    case "sell_answer":
                        aid, qty = request["answer_id"], float(request["qty"])
                        idx = self.answer_id_to_idx.get(aid)
                        if idx is not None and self.holdings[agent][aid] >= qty:
                            dq_by_agent[agent][idx] -= qty
                    case "buy_new_answer":
                        if not getattr(self, "fixed_answers", False):
                            ans_text, amount = (
                                request["answer_text"],
                                float(request["amount"]),
                            )
                            # Enforce minimum investment
                            if num_answers_at_start == 0 or amount >= min_investment:
                                if ans_text not in new_answers:
                                    new_answers[ans_text] = []
                                new_answers[ans_text].append((agent, amount))
                    case "buy_message":
                        msg = request["msg"]
                        price = self.message_price()
                        if self.balances[agent] >= price:
                            self.balances[agent] -= price
                            new_msgs.append((agent, msg))

        # --- Create all new answers first ---
        new_answer_ids = {}
        if not getattr(self, "fixed_answers", False):
            for ans_text in new_answers:
                new_id = self._add_new_answer(ans_text, dq_by_agent)
                new_answer_ids[ans_text] = new_id
            # Add buy requests for new answers to the main buy_requests list
            for ans_text, contribs in new_answers.items():
                aid = new_answer_ids[ans_text]
                for agent, amount in contribs:
                    # Set the agent's holding and market cap to the amount paid
                    idx = self.answer_id_to_idx[aid]
                    dq_by_agent[agent][idx] += amount

        # --- Process all buy requests together ---
        for agent, aid, amount in buy_requests:
            idx = self.answer_id_to_idx.get(aid)
            if idx is not None:
                max_qty = self._max_shares_for_amount(idx, amount, False)
                dq_by_agent[agent][idx] += max_qty

        # --- Batch clearing trades ---
        total_dq = np.zeros_like(self.q)
        for dq in dq_by_agent.values():
            total_dq += dq

        for agent, dq in dq_by_agent.items():
            if not np.any(dq):
                continue
            indiv_cost = self.cost(self.q + dq) - self.cost(self.q)
            if self.balances[agent] >= indiv_cost:
                self.balances[agent] -= indiv_cost
                # Update holdings by answer_id
                for aid, idx in self.answer_id_to_idx.items():
                    self.holdings[agent][aid] += dq[idx]
            else:
                # skip if insufficient balance
                pass

        self.q += total_dq

        # Record messages (this round only + full history)
        self.messages_round.extend(new_msgs)
        self.messages_all.extend(new_msgs)
        self.msg_counts.append(len(new_msgs))

        # Remove answers with zero holdings only if not fixed_answers
        if not getattr(self, "fixed_answers", False):
            to_remove = []
            for idx, aid in enumerate(self.answer_ids):
                total_holdings = sum(self.holdings[ag][aid] for ag in self.agents)
                if total_holdings == 0:
                    to_remove.append(idx)
            # Remove in reverse order to avoid index shift
            for idx in reversed(to_remove):
                self._remove_answer(idx)

        # Only include answers and holdings for currently active answer_ids
        prices = self.prices(self.q) if len(self.q) > 0 else np.array([])
        observations = {
            agent: {
                "balance": self.balances[agent],
                "holdings": {aid: self.holdings[agent][aid] for aid in self.answer_ids},
                "prices": prices.tolist(),
                "answers": [
                    {"id": aid, "text": self.answers[self.answer_id_to_idx[aid]]}
                    for aid in self.answer_ids
                ],
                "message_price": self.message_price(),
                "messages": self.messages_round,
            }
            for agent in self.agents
        }

        # At the end of the first step after reset, disable the zero-answers flag
        if getattr(self, "round_started_with_zero_answers", False):
            self.round_started_with_zero_answers = False

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.q is not None and len(self.q) > 0:
            print("Prices:", self.prices(self.q))
        print("Balances:", self.balances)
        print("Messages this round:", self.messages_round)

    def close(self):
        pass

    def state(self):
        return {
            "q": self.q.tolist() if self.q is not None else [],
            "answers": self.answers,
            "holdings": {k: v.tolist() for k, v in self.holdings.items()},
            "balances": self.balances,
            "messages_all": self.messages_all,
            "messages_round": self.messages_round,
            "recent_msg_counts": list(self.msg_counts),
        }
