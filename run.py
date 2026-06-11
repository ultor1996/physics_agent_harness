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

# Only the 6 recoverable parameters
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


def build_task(inp: dict) -> str:
    dp = inp["data_paths"]
    return f"""Analyse this gravitational-wave merger event and return all 6 parameters.

Task ID:     {inp['task_id']}
Approximant: {inp.get('approximant', 'IMRPhenomD')}
Sample rate: {inp.get('sample_rate_hz', 2048)} Hz
f_lower:     {inp.get('f_lower_hz', 20.0)} Hz

File paths:
  strain_H1:  {dp['strain_H1']}
  strain_L1:  {dp['strain_L1']}
  psd_H1:     {dp['psd_H1']}
  psd_L1:     {dp['psd_L1']}
  psd_freqs:  {dp['psd_freqs']}
  times:      {dp['times']}

Instructions:
  1. Call load_gw_data with the 6 paths above
  2. Call matched_filter_chirp_mass on H1 strain using approximant "{inp.get('approximant', 'IMRPhenomD')}"
  3. Call estimate_component_masses with the best_chirp_mass_Msun
  4. Call classify_merger_type
  5. Call final_answer with all 6 parameters
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

    agent = create_gw_agent(make_model())
    task  = build_task(inp)

    # Log agent interaction
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

    # Parse result
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