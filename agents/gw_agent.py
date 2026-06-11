import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools import (
    load_gw_data,
    matched_filter_chirp_mass,
    estimate_component_masses,
    classify_merger_type,
)

def create_gw_agent(model, log_path: str = None):
    prompt_path = Path(__file__).parent.parent / "prompts" / "gw_agent.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[
            load_gw_data,
            matched_filter_chirp_mass,
            estimate_component_masses,
            classify_merger_type,
        ],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=5,
        additional_authorized_imports=[
            "numpy", "pycbc", "json", "math"
        ],
    )