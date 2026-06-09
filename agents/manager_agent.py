# agents/manager_agent.py
import yaml
from smolagents import CodeAgent


def load_prompt(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_manager_agent(model, managed_agents: list):
    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=managed_agents,
        name="manager",
        description="Coordinates research and data analysis tasks.",
        instructions="Always delegate research tasks to research_agent and data tasks to data_agent. Never answer from memory.",
        max_steps=5,
    )