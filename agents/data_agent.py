import yaml
from smolagents import CodeAgent
from tools.custom_tools import analyze_dataframe


def load_prompt(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_data_agent(model):
    return CodeAgent(
        tools=[
            analyze_dataframe,
        ],
        model=model,
        additional_authorized_imports=["pandas", "csv"],
        prompt_templates=load_prompt("prompts/data_agent.yaml"),
        name="data_agent",
        description="Analyzes CSV data files and answers questions about them. Input: a data analysis task with a file path.",
        max_steps=5,    # default is 20 — lower means fewer LLM calls but may not finish complex tasks
    )