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

from agents.planning_agent_v2 import create_planning_agent
from agents.execution_agent_v2 import create_execution_agent
from agents.critic_agent_v2 import create_critic_agent
from agents.report_agent_v2 import create_report_agent

SAFE_DEFAULTS = {"network_snr": 0.0, "coalescence_time_s": 0.0}
REQUIRED_KEYS = [
    "chirp_mass_Msun", "mass1_Msun", "mass2_Msun",
    "mass_ratio", "network_snr", "merger_type",
    "coalescence_time_s",
]
TOTAL_TIME_BUDGET_S = 3800  # overall budget for this task, tune as needed
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
Step 3 — call list_approximant_capabilities to see what physics is
available.
Step 4 — decide your own prior_spec: for chirp_mass, mass_ratio, a_1,
a_2, and theta_jn, decide whether to fix or sample each. If using a
precession-capable approximant (check list_approximant_capabilities),
you may also decide to sample tilt_1, tilt_2, phi_12, and phi_jl. Base
your decisions on the matched-filter seed's SNR and standard GW PE
practice. Call build_full_priors(approximant, prior_spec, rationale)
with your decision.
Step 5 — decide your sampler settings (nlive, sample, walks, nact, dlogz)
based on the seed, your time budget, AND how many dimensions you chose to
free in Step 4 (more free parameters generally need more live points to
converge reliably). Call build_pe_config(...) with your decision.
Step 6 — call final_answer with exactly:
{{
    "data": <dict from load_gw_data>,
    "mf": <dict from seed_pe_prior_via_matched_filter>,
    "priors_package": <dict from build_full_priors>,
    "config": <dict from build_pe_config>,
}}
"""


def build_execution_task(task: dict) -> str:
    given = task["given_parameters"]
    return f"""Run full Bayesian parameter estimation with the exact inputs,
priors, and config provided in additional_args. Do not modify the
priors_package or config, or choose your own settings.

Task ID:     {task['task_id']}
Approximant hint: {task['approximant_hint']}
f_lower:     {task['f_lower']} Hz

Known sky location (do NOT estimate these, use exactly as given):
  ra:            {given['ra']}
  dec:           {given['dec']}
  polarisation:  {given['polarisation']}

Call run_bayesian_pe using data, mf, priors_package, and config from
additional_args:
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
  approximant      = priors_package["approximant"]
  priors_package   = priors_package
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
pe_result["mass_ratio"], pe_result["coalescence_time_s"],
pe_result["a1_recovered"], pe_result["a2_recovered"],
pe_result["tilt1_recovered"], pe_result["tilt2_recovered"],
pe_result["phi12_recovered"], pe_result["phi_jl_recovered"], and
pe_result["theta_jn_recovered"] from additional_args (also uses
data["strain_H1"], data["psd_H1"], data["psd_freqs"], data["sample_rate"]
from additional_args).
Step 3 — reason about log_bayes_factor and chi2_reduced together. Also
consider fix_spins_used / fix_inclination_used from inspect_pe_result as
structural risk factors -- if spin or inclination was fixed at zero but
the fit still looks poor, that may indicate the wrong simplification was
made upstream during planning, not a sampler convergence problem.
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
        print("Usage: python run_gw_multi.py <input.json>")
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
    priors_package = None
    execution_result = None
    critique = None
    try:
        # ---- planning ----
        print(f"\n{'─'*70}\n[STAGE 1/4] PLANNING AGENT — starting\n{'─'*70}\n")
        planning_agent = create_planning_agent(model)
        planning_task_str = build_planning_task(task, time_budget_s=remaining_budget())
        planning_result = planning_agent.run(
            planning_task_str,
            additional_args={"data_paths": dp, "sample_rate": task["sample_rate"]},
        )
        data = planning_result["data"]
        mf = planning_result["mf"]
        priors_package = planning_result["priors_package"]
        config = planning_result["config"]
        print(f"\n[STAGE 1/4] PLANNING AGENT — done.\n  priors_package={priors_package}\n  config={config}\n")
        attempt = 0
        execution_result, critique = None, None
        while attempt <= MAX_RETRIES:
            # ---- PE execution ----
            print(f"\n{'─'*70}\n[STAGE 2/4] EXECUTION AGENT — starting \n{'─'*70}\n")
            execution_agent = create_execution_agent(model)
            execution_task_str = build_execution_task(task)   # now includes given_ra/dec/psi internally
            execution_result = execution_agent.run(
                execution_task_str,
                additional_args={"data": data, "mf": mf, "priors_package": priors_package, "config": config},
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
            # NOTE: retries currently only adjust sampler settings (config),
            # not priors_package. If you want the critic to be able to
            # request a different prior_spec on retry (e.g. "free spin
            # after all"), that needs a new retry_prior_spec field on
            # package_pe_critique and handling here -- not yet implemented.
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

    diagnostics_path = Path("/home/sr/Desktop/code/GW_merger_bench/results/diagnostics") / f"{task['task_id']}.json"
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "priors_package": priors_package,
        "pe_result": execution_result,
        "critique": critique,
    }
    try:
        with open(diagnostics_path, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)
    except Exception as e:
        print(f"[orchestrator] failed to write diagnostics: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()