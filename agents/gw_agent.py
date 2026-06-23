import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools import (
    load_gw_data,
    seed_pe_prior_via_matched_filter,
    run_bayesian_pe,
    classify_merger_type,
    plot_chirp_signal,
)

def create_gw_agent(model, log_path: str = None):
    prompt_path = Path(__file__).parent.parent / "prompts" / "gw_agent.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[
            load_gw_data,
            seed_pe_prior_via_matched_filter,
            run_bayesian_pe,
            classify_merger_type,
            plot_chirp_signal,
        ],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=8,
        planning_interval=None,
        additional_authorized_imports=[
    "numpy", "pycbc", "json", "math",
    "gwpy", "gwpy.timeseries",
    "scipy", "scipy.signal",
    "matplotlib", "matplotlib.pyplot",
    "bilby", "os", "pathlib",
    "logging", "warnings",
],
        executor_kwargs={"timeout_seconds": 1700},    
    )