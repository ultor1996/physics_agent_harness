import yaml
from smolagents import CodeAgent, WebSearchTool, VisitWebpageTool


def load_prompt(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_research_agent(model):
    return CodeAgent(
        tools=[
            WebSearchTool(),
            VisitWebpageTool(),
        ],
        model=model,
        prompt_templates=load_prompt("prompts/research_agent.yaml"),
        name="research_agent",
        description="Researches topics on the web and analyzes data. Input: a research question or analysis task.",
        max_steps=5,    # default is 20 — lower means fewer LLM calls but may not finish complex tasks
    )