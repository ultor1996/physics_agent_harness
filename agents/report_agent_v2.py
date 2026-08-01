import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools_v2 import (
    classify_merger_type,
    plot_chirp_signal,
)

def create_report_agent(model):
    prompt_path = Path(__file__).parent.parent / "prompts" / "report_agent_v2.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[
            classify_merger_type,
            plot_chirp_signal,
        ],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=None,
        additional_authorized_imports=[
            "numpy", "pycbc", "json", "math",
        ],
        executor_kwargs={"timeout_seconds": 120},
    )