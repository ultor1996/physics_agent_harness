import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools_v2 import (
    load_gw_data,
    seed_pe_prior_via_matched_filter,
    list_approximant_capabilities,
    build_full_priors,
    build_pe_config,
)

def create_planning_agent(model):
    prompt_path = Path(__file__).parent.parent / "prompts" / "planning_agent_v2.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[
            load_gw_data,
            seed_pe_prior_via_matched_filter,
            list_approximant_capabilities,
            build_full_priors,
            build_pe_config,
        ],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=None,
        additional_authorized_imports=[
            "numpy", "pycbc", "json", "math",
        ],
        executor_kwargs={"timeout_seconds": 300},
    )