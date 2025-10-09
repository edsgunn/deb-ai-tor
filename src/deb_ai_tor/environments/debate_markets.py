import json
import math
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
        return self.b * math.log(np.exp(q / self.b).sum())

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
        self.q = np.zeros(len(self.answers), dtype=float)
        self.holdings = {
            agent: np.zeros(len(self.answers), dtype=float) for agent in self.agents
        }
        self.balances = {agent: 1.0 for agent in self.agents}
        self.messages_all = []
        self.messages_round = []
        self.msg_counts.clear()

        self.answer_ids = [str(uuid.uuid4()) for _ in self.answers]
        self.answer_id_to_idx = {aid: i for i, aid in enumerate(self.answer_ids)}

        observations = [question for _ in self.agents]
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _add_new_answer(self, ans_text):
        self.answers.append(ans_text)
        new_id = str(uuid.uuid4())
        self.answer_ids.append(new_id)
        self.answer_id_to_idx[new_id] = len(self.answers) - 1
        self.q = np.append(self.q, 0.0)
        for ag in self.agents:
            self.holdings[ag] = np.append(self.holdings[ag], 0.0)
        return new_id

    def _max_shares_for_amount(self, idx, amount):
        # Binary search for max shares that can be bought for 'amount' on answer idx
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
        return left

    def _remove_answer(self, idx):
        # Remove answer at idx from all relevant structures
        del self.answers[idx]
        aid = self.answer_ids[idx]
        del self.answer_ids[idx]
        del self.answer_id_to_idx[aid]
        self.q = np.delete(self.q, idx)
        for ag in self.agents:
            self.holdings[ag] = np.delete(self.holdings[ag], idx)
        # Rebuild id->idx mapping
        self.answer_id_to_idx = {aid: i for i, aid in enumerate(self.answer_ids)}

    def step(self, actions):
        self.step_count += 1
        rewards = {agent: 0.0 for agent in self.agents}
        self.messages_round = []

        dq_by_agent = {agent: np.zeros_like(self.q) for agent in self.agents}
        new_answers = defaultdict(list)  # text -> [(agent, qty)]
        new_msgs = []

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
                        aid, amount = request["answer_id"], float(request["amount"])
                        idx = self.answer_id_to_idx.get(aid)
                        if idx is not None:
                            max_qty = self._max_shares_for_amount(idx, amount)
                            dq_by_agent[agent][idx] += max_qty

                    case "sell_answer":
                        aid, qty = request["answer_id"], float(request["qty"])
                        idx = self.answer_id_to_idx.get(aid)
                        if idx is not None and self.holdings[agent][idx] >= qty:
                            dq_by_agent[agent][idx] -= qty

                    case "buy_new_answer":
                        ans_text, amount = (
                            request["answer_text"],
                            float(request["amount"]),
                        )
                        new_answers[ans_text].append((agent, amount))

                    case "buy_message":
                        msg = request["msg"]
                        price = self.message_price()
                        if self.balances[agent] >= price:
                            self.balances[agent] -= price
                            new_msgs.append((agent, msg))

        # --- Handle new answers (deduplicate by text) ---
        for ans_text, contribs in new_answers.items():
            new_id = self._add_new_answer(ans_text)
            idx = self.answer_id_to_idx[new_id]
            for agent, amount in contribs:
                max_qty = self._max_shares_for_amount(idx, amount)
                dq_by_agent[agent][idx] += max_qty

        # --- Batch clearing trades ---
        total_dq = np.zeros_like(self.q)
        for dq in dq_by_agent.values():
            total_dq += dq

        # old_cost = self.cost(self.q)
        # new_cost = self.cost(self.q + total_dq)
        # total_cost = new_cost - old_cost

        for agent, dq in dq_by_agent.items():
            if not np.any(dq):
                continue
            indiv_cost = self.cost(self.q + dq) - self.cost(self.q)
            if self.balances[agent] >= indiv_cost:
                self.balances[agent] -= indiv_cost
                self.holdings[agent] += dq
            else:
                # skip if insufficient balance
                pass

        self.q += total_dq

        # Record messages (this round only + full history)
        self.messages_round.extend(new_msgs)
        self.messages_all.extend(new_msgs)
        self.msg_counts.append(len(new_msgs))

        # Build observations
        prices = self.prices(self.q) if len(self.q) > 0 else np.array([])
        observations = {
            agent: {
                "balance": self.balances[agent],
                "holdings": self.holdings[agent].tolist(),
                "prices": prices.tolist(),
                "answers": [
                    {"id": aid, "text": self.answers[self.answer_id_to_idx[aid]]}
                    for aid in self.answer_ids
                ],
                "message_price": self.message_price(),
                "messages": self.messages_round,  # only this round
            }
            for agent in self.agents
        }

        # After all trades, remove answers with zero holdings
        to_remove = []
        for idx, aid in enumerate(self.answer_ids):
            total_holdings = sum(self.holdings[ag][idx] for ag in self.agents)
            if total_holdings == 0:
                to_remove.append(idx)
        # Remove in reverse order to avoid index shift
        for idx in reversed(to_remove):
            self._remove_answer(idx)

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
