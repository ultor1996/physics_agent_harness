import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools import  plot_chirp_signal, classify_merger_type

def create_report_agent(model):
    prompt_path = Path(__file__).parent.parent / "prompts" / "report_agent.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[ plot_chirp_signal, classify_merger_type],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=None,
        additional_authorized_imports=[
            "numpy", "gwpy", "gwpy.timeseries",
            "scipy", "scipy.signal",
            "matplotlib", "matplotlib.pyplot", "os",
        ],
        executor_kwargs={"timeout_seconds": 120},
    )