# physics_agent_harness

Agentic gravitational-wave parameter estimation pipeline built on [smolagents](https://github.com/huggingface/smolagents). Designed as an external pipeline for [GW Merger Bench](https://github.com/your-username/GW_merger_bench).

## Prerequisites

- Python 3.10+
- Access to a LiteLLM-compatible endpoint
- pycbc, bilby, gwpy (installed via requirements)

## Installation

```bash
git clone https://github.com/your-username/physics_agent_harness.git
cd physics_agent_harness

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in:

```bash
OPENAI_API_BASE=http://your-litellm-host/v1
OPENAI_API_KEY=your_api_key_here
```

> `.env` is in `.gitignore` and will never be committed.

## Usage

The pipeline is called by the benchmark runner — it is not invoked directly. The benchmark passes an `input.json` path as the sole argument:

```bash
python run.py /path/to/input.json
```

`input.json` contains absolute paths to the strain/PSD `.npy` files, physics metadata, and an `output_path` where the agent must write its result.

### Running the unit tests

```bash
python unit_test_tools.py
```

Tests all 6 tools against a synthetic BBH injection with known parameters. Expected output: 29/29 passing.

## How the pipeline works

```
run.py
  ↓
build_task()      — constructs a 6-step task prompt with exact tool arguments
  ↓
create_gw_agent() — CodeAgent with 5 tools, planning disabled, max_steps=8
  ↓
agent.run(task)   — executes the 6-step pipeline in a single code block
  ↓
sanitise()        — validates output types and fills missing keys
  ↓
output.json       — written to the path specified in input.json
```

## Agent tools

| Tool | Purpose | Runtime |
|---|---|---|
| `load_gw_data` | Load 5 `.npy` files into a dict | ~1s |
| `seed_pe_prior_via_matched_filter` | 40×12 coarse matched filter grid — provides Mc seed and merger time | ~60s |
| `run_bayesian_pe` | Full Bilby + dynesty nested sampling PE | ~7–15 min |
| `classify_merger_type` | BBH / BNS / NSBH from component masses | <1s |
| `plot_chirp_signal` | 3-panel figure: whitened H1/L1 + Q-transform | ~10s |

The 6th "tool" is `final_answer` — a built-in smolagents tool that terminates the agent and returns the result dict.

### PE design

`run_bayesian_pe` follows the [avivajpeyi GW PE tutorial](https://avivajpeyi.github.io/gw_pe_tutorial/workshop_notebook.html) conventions:

- **Sampler**: dynesty nested sampling (`sample="slice"`, `nact=10`, `nlive=250`)
- **Sampled parameters**: `chirp_mass`, `mass_ratio`, `theta_jn` (3D)
- **Spins**: `a_1`, `a_2` fixed to `DeltaFunction(0.0)` for zero-spin dataset; switch to `Uniform(-spin_max, spin_max)` for spinning datasets
- **Sky/orientation**: `ra`, `dec`, `psi` fixed to DeltaFunction — unreliable without timing delay
- **Marginalised**: time, distance, phase analytically via `GravitationalWaveTransient`
- **Time reference**: `start_time = -merger_time_s` shifts the data so the matched-filter SNR peak is near `geocent_time=0`, ensuring the ±0.1s marginalisation window covers the actual merger

### Approximant-aware spin limits

```python
spin_limits = {
    "IMRPhenomD":   0.88,
    "IMRPhenomXHM": 0.99,
    "SEOBNRv4":     0.99,
}
```

When sampling spins, the prior range is automatically set from this dict based on the `approximant` argument passed by the benchmark.

## Agent configuration

`agents/gw_agent.py`:

```python
CodeAgent(
    tools=[load_gw_data, seed_pe_prior_via_matched_filter,
           run_bayesian_pe, classify_merger_type, plot_chirp_signal],
    model=model,
    prompt_templates=prompt_templates,
    max_steps=8,
    planning_interval=None,    # planning disabled — fixed 6-step pipeline
    additional_authorized_imports=[
        "numpy", "pycbc", "json", "math",
        "gwpy", "gwpy.timeseries",
        "scipy", "scipy.signal",
        "matplotlib", "matplotlib.pyplot",
        "bilby", "os", "pathlib", "logging", "warnings",
    ],
    executor_kwargs={"timeout_seconds": 1700},
)
```

`planning_interval=None` is critical — setting it to any integer causes the agent to fire a planning step at step 1 (due to `(step-1) % interval == 0`), which wastes tokens and confuses the agent since its tools are not visible in the planning context.

## Output

The agent writes `output.json` to the path specified in `input.json`:

```json
{
    "chirp_mass_Msun": 28.04,
    "mass1_Msun":      35.47,
    "mass2_Msun":      28.75,
    "mass_ratio":      0.810,
    "network_snr":     23.09,
    "merger_type":     "BBH"
}
```

Plots are saved to `GW_merger_bench/results/plots/{task_id}_chirp.png`.
Agent logs are saved to `GW_merger_bench/results/agent_logs/{task_id}_agent.log`.

## Repo structure

```
physics_agent_harness/
├── agents/
│   └── gw_agent.py          — CodeAgent with 5 GW tools
├── prompts/
│   └── gw_agent.yaml        — system prompt + final_answer fallback
├── tools/
│   └── gw_tools.py          — 6 tools (load, matched filter, PE, classify, plot)
├── run.py                   — benchmark entry point
├── unit_test_tools.py       — standalone tool unit tests (29 tests)
└── .env.example
```

## Debugging

Common issues and fixes:

| Error | Fix |
|---|---|
| `Import of os is not allowed` | Add `"os", "pathlib"` to `additional_authorized_imports` |
| `planning fires at step 1` | Set `planning_interval=None` |
| `FileNotFoundError` on plot output | `plot_chirp_signal` creates the directory internally via `os.makedirs` |
| `ZeroDivisionError` in smolagents | `planning_interval=0` — set to `None` instead |
| Slice sampler warning from dynesty | Cosmetic only — add `warnings.filterwarnings("ignore", message="Specifying slice option")` |