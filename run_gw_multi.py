"""
Multi-agent orchestrator for GW-Merger-Bench.
Replaces the old single-agent run.py.
Called by the benchmark as:
    python run_gw_multi.py <path_to_input.json>
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from smolagents import LiteLLMModel

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from agents.planning_agent import create_planning_agent
from agents.execution_agent import create_execution_agent
from agents.critic_agent import create_critic_agent
from agents.report_agent import create_report_agent

SAFE_DEFAULTS = {"network_snr": 0.0, "coalescence_time_s": 0.0}
REQUIRED_KEYS = [
    "chirp_mass_Msun", "mass1_Msun", "mass2_Msun",
    "mass_ratio", "network_snr", "merger_type",
    "coalescence_time_s",
]
TOTAL_TIME_BUDGET_S = 3800 # overall budget for this task, tune as needed
MAX_RETRIES = 1              # cap retries so a stuck critic can't loop forever


def make_model():
    return LiteLLMModel(
        model_id="openai/gpt-5.4-2026-03-05",
        api_base=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---- task builders ----

def build_planning_task(task: dict, time_budget_s: int) -> str:
 return f"""Triage this gravitational-wave event before full parameter estimation.

Task ID:     {task['task_id']}
Approximant hint: {task['approximant_hint']}
Sample rate: {task['sample_rate']} Hz
Segment duration: {task['segment_duration']} s
f_lower:     {task['f_lower']} Hz
Detectors:   {', '.join(task['detectors'])}
Time budget for the full PE step: {time_budget_s} seconds

Step 1 — call load_gw_data with the strain/PSD file paths for this task.
Step 2 — call seed_pe_prior_via_matched_filter using the loaded data.
Step 3 — call build_pe_config yourself, deciding nlive, sample, walks,
nact, dlogz, fix_spins, fix_inclination based on the matched-filter seed
and the time budget above.
Step 4 — call final_answer with exactly:
{{
    "data": <dict from load_gw_data>,
    "mf": <dict from seed_pe_prior_via_matched_filter>,
    "config": <dict from build_pe_config>,
}}
"""


def build_execution_task(task: dict) -> str:
    given = task["given_parameters"]
    return f"""Run full Bayesian parameter estimation with the exact inputs and
config provided in additional_args. Do not modify the config or choose
your own settings.

Task ID:     {task['task_id']}
Approximant: {task['approximant_hint']}
f_lower:     {task['f_lower']} Hz

Known sky location (do NOT estimate these, use exactly as given):
  ra:            {given['ra']}
  dec:           {given['dec']}
  polarisation:  {given['polarisation']}

Call run_bayesian_pe using data, mf, and config from additional_args:
  strain_H1        = data["strain_H1"]
  psd_H1           = data["psd_H1"]
  strain_L1        = data["strain_L1"]
  psd_L1           = data["psd_L1"]
  psd_freqs        = data["psd_freqs"]
  sample_rate      = data["sample_rate"]
  chirp_mass_guess = mf["best_chirp_mass_Msun"]
  mass_ratio_guess = mf["best_mass_ratio"]
  merger_time_s    = mf["merger_time_s"]
  given_ra         = {given['ra']}
  given_dec        = {given['dec']}
  given_psi        = {given['polarisation']}
  f_lower          = {task['f_lower']}
  approximant      = "{task['approximant_hint']}"
  config           = config

This may take many minutes. Do not interrupt it or attempt a faster
substitute. Call final_answer with the run_bayesian_pe result, unmodified.
"""


def build_critic_task(remaining_budget_s: int) -> str:
    return f"""Critique the PE result provided in additional_args and decide
whether to accept it or recommend a retry, given the remaining time budget.
You have no access to the true parameter values, so reason from posterior
diagnostics, known structural risk factors, and an independent waveform
residual check only.

Remaining time budget: {remaining_budget_s} seconds

Step 1 — call inspect_pe_result with pe_result from additional_args.
Step 2 — call check_waveform_residual using pe_result["chirp_mass_Msun"],
pe_result["mass_ratio"], and pe_result["coalescence_time_s"] from
additional_args (also uses data["strain_H1"], data["psd_H1"],
data["psd_freqs"], data["sample_rate"] from additional_args).
Step 3 — reason about log_bayes_factor and chi2_reduced together.
Step 4 — call package_pe_critique with your own recommendation and
reasoning.

Note: package_pe_critique will reject recommendation="accept" if
chi2_reduced > 3.0 or log_bayes_factor < 0. In that case, use
recommendation="retry" with adjusted sampler settings, or
"accept_with_caveat" with an explicit budget-based justification.

Call final_answer with package_pe_critique's exact result.
"""


def build_report_task(task: dict, plot_path: str) -> str:
    return f"""Produce the final classification, plot, and 6-key answer for
this event, using pe_result and mf provided in additional_args. Do not
re-run PE or second-guess these numbers.

Task ID: {task['task_id']}

Step 1 — call classify_merger_type using pe_result["mass1_Msun"] and
pe_result["mass2_Msun"].
Step 2 — call plot_chirp_signal, saving to "{plot_path}", using data
and mf from additional_args.
Step 3 — call final_answer with exactly:
{{
    "chirp_mass_Msun":    float(pe_result["chirp_mass_Msun"]),
    "mass1_Msun":         float(pe_result["mass1_Msun"]),
    "mass2_Msun":         float(pe_result["mass2_Msun"]),
    "mass_ratio":         float(pe_result["mass_ratio"]),
    "network_snr":        float(mf["best_snr"]),
    "coalescence_time_s": float(pe_result["coalescence_time_s"]),
    "merger_type":        <classify_merger_type's result>,
}}
merger_type must be exactly "BBH", "BNS", or "NSBH"
"""


def sanitise(raw) -> dict:
    if not isinstance(raw, dict):
        raw = {}
    out = {}
    for k in REQUIRED_KEYS:
        val = raw.get(k, SAFE_DEFAULTS.get(k, 0.0))
        if k == "merger_type":
            val = str(val).strip().upper()
            if val not in ("BBH", "BNS", "NSBH"):
                val = "BBH"
        else:
            try:
                val = float(val)
                if val != val:
                    val = float(SAFE_DEFAULTS.get(k, 0.0))
            except (TypeError, ValueError):
                val = float(SAFE_DEFAULTS.get(k, 0.0))
        out[k] = val
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_gw_multiagent.py <input.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        inp = json.load(f)

    output_path = inp["output_path"]

    # Locate this task's directory from data_paths (still needed to find task.json itself)
    task_dir = Path(inp["data_paths"]["strain_H1"]).parent
    with open(task_dir / "task.json") as f:
        task = json.load(f)

    # Resolve absolute paths using task.json's own data_files + this directory
    dp = {k: str(task_dir / v) for k, v in task["data_files"].items()}
    print(f"\n{'='*70}\n[ORCHESTRATOR] Starting task {task['task_id']}\n{'='*70}\n")
    plots_dir = Path("/home/sr/Desktop/code/GW_merger_bench/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("/home/sr/Desktop/code/GW_merger_bench/results/agent_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    model = make_model()
    start_time = time.time()

    def remaining_budget():
        return max(0, int(TOTAL_TIME_BUDGET_S - (time.time() - start_time)))

    final_result = {}
    try:
        # ---- planning ----
        print(f"\n{'─'*70}\n[STAGE 1/4] PLANNING AGENT — starting\n{'─'*70}\n")
        planning_agent = create_planning_agent(model)
        planning_task_str = build_planning_task(task, time_budget_s=remaining_budget())
        planning_result = planning_agent.run(
            planning_task_str,
            additional_args={"data_paths": dp, "sample_rate": task["sample_rate"]},
        )
        data, mf, config = planning_result["data"], planning_result["mf"], planning_result["config"]
        print(f"\n[STAGE 1/4] PLANNING AGENT — done. config={config}\n")
        attempt = 0
        execution_result, critique = None, None
        while attempt <= MAX_RETRIES:
            # ---- PE execution ----
            print(f"\n{'─'*70}\n[STAGE 2/4] EXECUTION AGENT — starting \n{'─'*70}\n")
            execution_agent = create_execution_agent(model)
            execution_task_str = build_execution_task(task)   # now includes given_ra/dec/psi internally
            execution_result = execution_agent.run(
                execution_task_str,
                additional_args={"data": data, "mf": mf, "config": config},
            )
            print(f"\n[STAGE 2/4] EXECUTION AGENT — done. coalescence_time={execution_result.get('coalescence_time_s')}, chirp_mass={execution_result.get('chirp_mass_Msun')}\n")

            # ---- Critic ----
            print(f"\n{'─'*70}\n[STAGE 3/4] CRITIC AGENT — starting\n{'─'*70}\n")
            critic_agent = create_critic_agent(model)
            critic_task_str = build_critic_task(remaining_budget_s=remaining_budget())
            critique = critic_agent.run(
                critic_task_str,
                additional_args={"pe_result": execution_result, "mf": mf, "data": data},
            )
            print(f"\n[STAGE 3/4] CRITIC AGENT — done. recommendation={critique.get('recommendation')}\n")

            if critique["recommendation"] != "retry" or attempt == MAX_RETRIES:
                break

            print(f"\n[ORCHESTRATOR] Critic recommended retry — updating config and looping back to PE\n")
            config = dict(config)
            if critique.get("retry_nlive"):
                config["nlive"] = critique["retry_nlive"]
            if critique.get("retry_dlogz"):
                config["dlogz"] = critique["retry_dlogz"]
            if critique.get("retry_sample"):
                config["sample"] = critique["retry_sample"]
            if critique.get("retry_walks"):
                config["walks"] = critique["retry_walks"]
            if critique.get("retry_nact"):
                config["nact"] = critique["retry_nact"]
            attempt += 1

        # ---- Reporting ----
        print(f"\n{'─'*70}\n[STAGE 4/4] REPORTING AGENT — starting\n{'─'*70}\n")
        plot_path = str(plots_dir / f"{task['task_id']}_chirp.png")
        report_agent = create_report_agent(model)
        report_task_str = build_report_task(task, plot_path)
        final_result = report_agent.run(
            report_task_str,
            additional_args={"pe_result": execution_result, "mf": mf, "data": data},
        )

    except Exception as e:
            print(f"[orchestrator] error: {e}", file=sys.stderr)

        
    safe_result = sanitise(final_result)
    try:
            with open(output_path, "w") as f:
                json.dump(safe_result, f, indent=2)
            print(f"[orchestrator] wrote output to {output_path}")
    except Exception as e:
            print(f"[orchestrator] FAILED to write output_path: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()