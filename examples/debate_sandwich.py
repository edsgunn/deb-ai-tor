import argparse
import json

from transformers import pipeline

from deb_ai_tor.environments import DebateSandwichEnv

parser = argparse.ArgumentParser(
    description="Run Debate Sandwich Environment with Local LLM Agents"
)
parser.add_argument(
    "--model-name",
    type=str,
    default="mistralai/Mistral-7B-Instruct-v0.2",
    help="Name of the local LLM model to use",
)
parser.add_argument(
    "--num-agents",
    type=int,
    default=2,
    help="Number of agents in the debate sandwich environment",
)
parser.add_argument("--num-rounds", type=int, default=3, help="Number of debate rounds")

args = parser.parse_args()

agents = [f"agent_{i}" for i in range(args.num_agents)]

env = DebateSandwichEnv(agents=agents)
obs = env.reset(
    options={
        "question": "What is a debate sandwich?",
        "answers": [
            "Arguments with answers in the middle",
            "A sandwich eaten while debating",
            "Answers with arguments in the middle",
        ],
        "num_rounds": args.num_rounds,
    }
)

print(json.dumps(obs, indent=4))


class LocalLLMAgent:
    def __init__(self, agent_name, llm_pipeline):
        self.agent_name = agent_name
        self.llm = llm_pipeline
        self.context = ""

    def get_action(self, obs):
        prompt = (
            f"Agent: {self.agent_name}\n"
            f"Previous context:\n{self.context}\n"
            f"Observation: {json.dumps(obs, indent=2)}\n"
            f"What is your action?"
        )
        response = self.llm(prompt, max_new_tokens=32)[0]["generated_text"]
        self.context += (
            f"\nObservation: {json.dumps(obs, indent=2)}\nAction: {response.strip()}"
        )
        return response.strip()


# Shared LLM pipeline for all agents
llm_pipeline = pipeline("text-generation", model=args.model_name, device_map="auto")

# Create agent wrappers
agent_wrappers = {name: LocalLLMAgent(name, llm_pipeline) for name in agents}

# Run a round using LLM agents
for _round in range(args.num_rounds):
    actions = {name: agent_wrappers[name].get_action(obs[name]) for name in agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print(json.dumps(obs, indent=4))
