import json

import pytest

from deb_ai_tor.environments.debate_markets import DebateMarket


@pytest.fixture()
def market():
    agents = ["A0", "A1", "A2"]
    return DebateMarket(agents=agents)


def test_buy_new_answer_gives_holdings_and_decreases_balance(market):
    market.reset(question="Test?")
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": f"{agent} ans", "amount": 0.5}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    for agent in market.agents:
        # Should have holdings in their own answer
        own_id = next(
            a["id"] for a in obs[agent]["answers"] if a["text"] == f"{agent} ans"
        )
        assert obs[agent]["holdings"][own_id] > 0
        # Balance should decrease
        assert obs[agent]["balance"] < 1.0


def test_sell_answer_removes_holdings(market):
    market.reset(question="Test?")
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": f"{agent} ans", "amount": 0.5}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    # Sell all holdings for one agent
    agent = market.agents[0]
    own_id = next(a["id"] for a in obs[agent]["answers"] if a["text"] == f"{agent} ans")
    qty = obs[agent]["holdings"][own_id]
    actions = {
        agent: json.dumps([{"type": "sell_answer", "answer_id": own_id, "qty": qty}])
    }
    obs, *_ = market.step(actions)
    # If the answer is still present, holdings should be zero; otherwise, it should be removed
    if own_id in obs[agent]["holdings"]:
        assert obs[agent]["holdings"][own_id] == 0
    else:
        assert own_id not in obs[agent]["holdings"]


def test_answer_disappears_when_no_holdings(market):
    market.reset(question="Test?")
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": f"{agent} ans", "amount": 0.5}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    # Sell all holdings for all agents
    sell_actions = {}
    for agent in market.agents:
        own_id = next(
            a["id"] for a in obs[agent]["answers"] if a["text"] == f"{agent} ans"
        )
        qty = obs[agent]["holdings"][own_id]
        sell_actions[agent] = json.dumps(
            [{"type": "sell_answer", "answer_id": own_id, "qty": qty}]
        )
    obs, *_ = market.step(sell_actions)
    # All answers should be gone
    for agent in market.agents:
        assert obs[agent]["answers"] == []


def test_buy_answer_by_amount(market):
    market.reset(question="Test?", answers=["foo", "bar"])
    # Get answer ids
    obs, *_ = market.step({})
    foo_id = next(a["id"] for a in obs["A0"]["answers"] if a["text"] == "foo")
    actions = {
        "A0": json.dumps([{"type": "buy_answer", "answer_id": foo_id, "amount": 0.5}])
    }
    obs, *_ = market.step(actions)
    assert obs["A0"]["holdings"][foo_id] > 0
    assert obs["A0"]["balance"] < 1.0


def test_fixed_answers_cannot_be_added_or_removed(market):
    # Provide fixed answers
    obs, _ = market.reset(question="Test?", answers=["foo", "bar"])
    # Try to add a new answer
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": f"{agent} ans", "amount": 0.5}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    # No new answers should be present
    for agent in market.agents:
        answer_texts = [a["text"] for a in obs[agent]["answers"]]
        assert set(answer_texts) == {"foo", "bar"}
    # Sell all holdings for all agents (should not remove answers)
    foo_id = next(a["id"] for a in obs["A0"]["answers"] if a["text"] == "foo")
    bar_id = next(a["id"] for a in obs["A0"]["answers"] if a["text"] == "bar")
    sell_actions = {
        agent: json.dumps(
            [
                {
                    "type": "sell_answer",
                    "answer_id": foo_id,
                    "qty": obs[agent]["holdings"][foo_id],
                },
                {
                    "type": "sell_answer",
                    "answer_id": bar_id,
                    "qty": obs[agent]["holdings"][bar_id],
                },
            ]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(sell_actions)
    # Answers should still be present
    for agent in market.agents:
        answer_texts = [a["text"] for a in obs[agent]["answers"]]
        assert set(answer_texts) == {"foo", "bar"}
        # Holdings should be zero
        assert obs[agent]["holdings"][foo_id] == 0
        assert obs[agent]["holdings"][bar_id] == 0


def test_cannot_buy_new_answer_when_fixed(market):
    obs, _ = market.reset(question="Test?", answers=["foo", "bar"])
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": "baz", "amount": 1.0}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    for agent in market.agents:
        answer_texts = [a["text"] for a in obs[agent]["answers"]]
        assert "baz" not in answer_texts


def test_buy_new_answer_when_not_fixed(market):
    market.reset(question="Test?")
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": "baz", "amount": 1.0}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    for agent in market.agents:
        answer_texts = [a["text"] for a in obs[agent]["answers"]]
        assert "baz" in answer_texts


def test_answer_not_removed_if_any_agent_holds(market):
    market.reset(question="Test?", fixed_answers=False)
    # Agent 0 creates a new answer
    actions = {
        "A0": json.dumps(
            [{"type": "buy_new_answer", "answer_text": "A0 ans", "amount": 0.5}]
        ),
        "A1": json.dumps([]),
        "A2": json.dumps([]),
    }
    obs, *_ = market.step(actions)
    own_id = next(a["id"] for a in obs["A0"]["answers"] if a["text"] == "A0 ans")
    # Agent 1 buys shares in Agent 0's answer
    actions = {
        "A1": json.dumps([{"type": "buy_answer", "answer_id": own_id, "amount": 0.5}])
    }
    obs, *_ = market.step(actions)
    # Agent 0 sells all holdings
    qty = obs["A0"]["holdings"][own_id]
    actions = {
        "A0": json.dumps([{"type": "sell_answer", "answer_id": own_id, "qty": qty}])
    }
    obs, *_ = market.step(actions)
    # Answer should still be present for Agent 1
    assert own_id in obs["A1"]["holdings"]
    assert obs["A1"]["holdings"][own_id] > 0


def test_answer_removed_when_all_agents_sell(market):
    market.reset(question="Test?")
    actions = {
        agent: json.dumps(
            [{"type": "buy_new_answer", "answer_text": "baz", "amount": 1.0}]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(actions)
    baz_id = next(a["id"] for a in obs["A0"]["answers"] if a["text"] == "baz")
    sell_actions = {
        agent: json.dumps(
            [
                {
                    "type": "sell_answer",
                    "answer_id": baz_id,
                    "qty": obs[agent]["holdings"][baz_id],
                }
            ]
        )
        for agent in market.agents
    }
    obs, *_ = market.step(sell_actions)
    for agent in market.agents:
        assert baz_id not in obs[agent]["holdings"]
        assert all(a["id"] != baz_id for a in obs[agent]["answers"])
