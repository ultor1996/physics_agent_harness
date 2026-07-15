import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools import run_bayesian_pe

def create_execution_agent(model):
    prompt_path = Path(__file__).parent.parent / "prompts" / "execution_agent.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[run_bayesian_pe],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=None,
        additional_authorized_imports=[
            "numpy", "bilby", "os", "pathlib", "logging", "warnings",
        ],
        executor_kwargs={"timeout_seconds": 3600},
    )