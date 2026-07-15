import yaml
from pathlib import Path
from smolagents import CodeAgent
from tools.gw_tools import inspect_pe_result, package_pe_critique, check_waveform_residual

def create_critic_agent(model):
    prompt_path = Path(__file__).parent.parent / "prompts" / "critic_agent.yaml"
    with open(prompt_path) as f:
        prompt_templates = yaml.safe_load(f)

    return CodeAgent(
        tools=[inspect_pe_result, package_pe_critique, check_waveform_residual],
        model=model,
        prompt_templates=prompt_templates,
        max_steps=5,
        planning_interval=None,
        additional_authorized_imports=["numpy", "json"],
        executor_kwargs={"timeout_seconds": 120},
    )