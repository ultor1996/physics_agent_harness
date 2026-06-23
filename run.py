"""
External pipeline entry point for GW-Merger-Bench.
Called by the benchmark as:
    python run.py <path_to_input.json>
"""

import io
import os
import sys
import json
from pathlib import Path
from contextlib import redirect_stdout
from dotenv import load_dotenv
from smolagents import LiteLLMModel

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

SAFE_DEFAULTS = {
    "network_snr": 0.0,
}

REQUIRED_KEYS = [
    "chirp_mass_Msun", "mass1_Msun", "mass2_Msun",
    "mass_ratio", "network_snr", "merger_type",
]


def make_model():
    return LiteLLMModel(
        model_id="openai/gpt-5.4-2026-03-05",
        api_base=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# def build_task(inp: dict) -> str:
#     dp  = inp["data_paths"]
#     sr  = inp.get('sample_rate_hz', 2048)
#     app = inp.get('approximant', 'IMRPhenomD')
#     fl  = inp.get('f_lower_hz', 20.0)
#     return f"""Analyse this gravitational-wave merger event and return all 6 parameters.

# Task ID:     {inp['task_id']}
# Approximant: {app}
# Sample rate: {sr} Hz
# f_lower:     {fl} Hz

# Step 1 — call load_gw_data with these exact arguments:
#   strain_H1   = "{dp['strain_H1']}"
#   strain_L1   = "{dp['strain_L1']}"
#   psd_H1      = "{dp['psd_H1']}"
#   psd_L1      = "{dp['psd_L1']}"
#   psd_freqs   = "{dp['psd_freqs']}"
#   sample_rate = {sr}

# Step 2 — call matched_filter_chirp_mass with ONLY these arguments:
#   strain      = data["strain_H1"]
#   psd         = data["psd_H1"]
#   psd_freqs   = data["psd_freqs"]
#   strain_L1   = data["strain_L1"]
#   psd_L1      = data["psd_L1"]
#   sample_rate = data["sample_rate"]
#   approximant = "{app}"
#   f_lower     = {fl}

# Step 3 — call parameter_estimation to refine chirp mass and mass ratio:
#   strain_H1        = data["strain_H1"]
#   psd_H1           = data["psd_H1"]
#   strain_L1        = data["strain_L1"]
#   psd_L1           = data["psd_L1"]
#   psd_freqs        = data["psd_freqs"]
#   sample_rate      = data["sample_rate"]
#   chirp_mass_guess = mf["best_chirp_mass_Msun"]
#   mass_ratio_guess = mf["best_mass_ratio"]

# Step 4 — call estimate_component_masses:
#   chirp_mass_Msun  = pe["chirp_mass_Msun"]
#   mass_ratio_guess = pe["mass_ratio"]

# Step 5 — call classify_merger_type:
#   mass1_Msun = masses["mass1_Msun"]
#   mass2_Msun = masses["mass2_Msun"]

# Step 6 — call final_answer with this exact dict:
#   {{
#       "chirp_mass_Msun": float(pe["chirp_mass_Msun"]),
#       "mass1_Msun":      float(masses["mass1_Msun"]),
#       "mass2_Msun":      float(masses["mass2_Msun"]),
#       "mass_ratio":      float(pe["mass_ratio"]),
#       "network_snr":     float(mf["best_snr"]),
#       "merger_type":     cls["merger_type"],
#   }}
#   merger_type must be exactly "BBH", "BNS", or "NSBH"
# """

def build_task(inp: dict, plots_dir: str = "/tmp") -> str:
    dp  = inp["data_paths"]
    sr  = inp.get('sample_rate_hz', 2048)
    app = inp.get('approximant', 'IMRPhenomD')
    fl  = inp.get('f_lower_hz', 20.0)
    plot_path = os.path.join(plots_dir, f"{inp['task_id']}_chirp.png")
    return f"""Analyse this gravitational-wave merger event and return all 6 parameters.

Task ID:     {inp['task_id']}
Approximant: {app}
Sample rate: {sr} Hz
f_lower:     {fl} Hz

Step 1 — call load_gw_data with these exact arguments:
  strain_H1   = "{dp['strain_H1']}"
  strain_L1   = "{dp['strain_L1']}"
  psd_H1      = "{dp['psd_H1']}"
  psd_L1      = "{dp['psd_L1']}"
  psd_freqs   = "{dp['psd_freqs']}"
  sample_rate = {sr}

Step 2 — call seed_pe_prior_via_matched_filter with these exact arguments:
  strain      = data["strain_H1"]
  psd         = data["psd_H1"]
  psd_freqs   = data["psd_freqs"]
  strain_L1   = data["strain_L1"]
  psd_L1      = data["psd_L1"]
  sample_rate = data["sample_rate"]
  approximant = "{app}"
  f_lower     = {fl}

Step 3 — call run_bayesian_pe with these exact arguments:
  strain_H1         = data["strain_H1"]
  psd_H1            = data["psd_H1"]
  strain_L1         = data["strain_L1"]
  psd_L1            = data["psd_L1"]
  psd_freqs         = data["psd_freqs"]
  sample_rate       = data["sample_rate"]
  chirp_mass_guess  = mf["best_chirp_mass_Msun"]
  mass_ratio_guess  = mf["best_mass_ratio"]
  merger_time_s     = mf["merger_time_s"]
  f_lower           = {fl}
  approximant       = "{app}"
  nlive             = 250
  This runs full Bayesian parameter estimation and takes several minutes —
  do not interrupt it or attempt a faster substitute.
  This returns mass1_Msun and mass2_Msun directly — do not call any other
  tool to compute component masses.

Step 4 — call classify_merger_type with these exact arguments:
  mass1_Msun = pe["mass1_Msun"]
  mass2_Msun = pe["mass2_Msun"]

Step 5 — call plot_chirp_signal with these exact arguments:
  strain_H1        = data["strain_H1"]
  strain_L1        = data["strain_L1"]
  psd_H1           = data["psd_H1"]
  psd_freqs        = data["psd_freqs"]
  sample_rate      = data["sample_rate"]
  chirp_mass_Msun  = pe["chirp_mass_Msun"]
  mass_ratio       = pe["mass_ratio"]
  output_path      = "{plot_path}"
  f_lower          = {fl}
  approximant      = "{app}"

Step 6 — call final_answer with this exact dict:
  {{
      "chirp_mass_Msun": float(pe["chirp_mass_Msun"]),
      "mass1_Msun":      float(pe["mass1_Msun"]),
      "mass2_Msun":      float(pe["mass2_Msun"]),
      "mass_ratio":      float(pe["mass_ratio"]),
      "network_snr":     float(mf["best_snr"]),
      "merger_type":     cls["merger_type"],
  }}
  merger_type must be exactly "BBH", "BNS", or "NSBH"
"""


def sanitise(raw) -> dict:
    """Ensure output has all 6 required keys with valid types."""
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
                if val != val:  # NaN check
                    val = float(SAFE_DEFAULTS.get(k, 0.0))
            except (TypeError, ValueError):
                val = float(SAFE_DEFAULTS.get(k, 0.0))
        out[k] = val
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <input.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        inp = json.load(f)

    output_path = inp["output_path"]

    from agents.gw_agent import create_gw_agent

    plots_dir = Path("/home/sr/Desktop/code/GW_merger_bench/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    agent = create_gw_agent(make_model())
    task  = build_task(inp, plots_dir=str(plots_dir))

    log_dir = Path("/home/sr/Desktop/code/GW_merger_bench/results/agent_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{inp['task_id']}_agent.log"

    buf    = io.StringIO()
    result = {}
    try:
        with redirect_stdout(buf):
            result = agent.run(task)
    except Exception as e:
        print(f"[gw_agent] agent error: {e}", file=sys.stderr)
    finally:
        log_content = buf.getvalue()
        print(log_content)
        with open(log_file, "w") as f:
            f.write(log_content)

    if isinstance(result, dict):
        raw = result
    else:
        import ast
        try:
            raw = json.loads(str(result))
        except Exception:
            try:
                raw = ast.literal_eval(str(result))
            except Exception:
                raw = {}

    output = sanitise(raw)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[gw_agent] written → {output_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()