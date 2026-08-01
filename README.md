# physics_agent_harness

Agentic gravitational-wave parameter estimation pipeline built on
[smolagents](https://github.com/huggingface/smolagents). Designed as an
external pipeline for
[GW Merger Bench](https://github.com/your-username/GW_merger_bench).

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

The pipeline is called by the benchmark runner — it is not invoked directly.
The benchmark passes an `input.json` path as the sole argument:

```bash
python run_gw_multi_v2.py /path/to/input.json
```

`input.json` contains absolute paths to the strain/PSD `.npy` files, physics
metadata, and an `output_path` where the pipeline must write its result.

## How the pipeline works

This is **not** a single agent running a fixed tool sequence — it's four
separate agents, each with their own tool subset and system prompt, run in
strict order by the orchestrator:

```
run_gw_multi_v2.py
  ↓
[1/4] PLANNING AGENT   — loads data, gets a matched-filter seed, decides
                          its OWN prior specification (which parameters to
                          sample, with what bounds) and sampler settings
  ↓
[2/4] EXECUTION AGENT  — runs full Bayesian PE with the planning agent's
                          exact priors/config, unmodified
  ↓
[3/4] CRITIC AGENT     — inspects the PE result, runs an independent
                          waveform-residual chi² check, decides
                          accept / retry / accept_with_caveat
                          (retry loops back to [2/4] with adjusted
                          sampler settings, up to MAX_RETRIES times)
  ↓
[4/4] REPORTING AGENT  — classifies merger type, plots the result,
                          assembles the final answer dict
  ↓
sanitise() + output.json written
```

Each stage is a `smolagents.CodeAgent` with `planning_interval=None` (no
internal smolagents planning step — see Debugging below) and only the tools
relevant to its job.

## Agent tools

All tools live in `tools/gw_tools_v2.py`.

| Tool | Used by | Purpose |
|---|---|---|
| `load_gw_data` | Planning | Load strain/PSD `.npy` files into a dict |
| `seed_pe_prior_via_matched_filter` | Planning | Two-stage coarse matched-filter grid (30×30 + per-mass-ratio chirp-mass refinement) — provides `chirp_mass`/`mass_ratio`/SNR seed and merger time |
| `list_approximant_capabilities` | Planning | Returns which physics (precession, higher modes) each registered approximant supports, so the agent doesn't have to recall this from memory |
| `build_full_priors` | Planning | Validates and packages the agent's own prior specification — physical bounds, approximant compatibility, precession-angle gating |
| `build_pe_config` | Planning | Validates the agent's sampler settings (`nlive`, `sample`, `walks`, `nact`, `dlogz`) |
| `run_bayesian_pe` | Execution | Full Bilby + dynesty nested-sampling PE, built entirely from the planning agent's `priors_package`/`config` |
| `inspect_pe_result` | Critic | Objective diagnostics from a PE result (CI widths, log Bayes factor, which parameters were free) — does not decide anything |
| `check_waveform_residual` | Critic | Ground-truth-free chi² consistency test (Allen 2005) — independent signal of fit quality |
| `package_pe_critique` | Critic | Validates the critic's own accept/retry decision; blocks a plain "accept" if `chi2_reduced > 3.0` or `log_bayes_factor < 0` |
| `classify_merger_type` | Reporting | BBH / BNS / NSBH from recovered component masses |
| `plot_chirp_signal` | Reporting | 3-panel figure: whitened H1/L1 strain + best-fit template overlay + Q-transform |
| `estimate_component_masses` | (fallback) | Standalone `chirp_mass`+`q` → `m1`/`m2` conversion, only needed if `run_bayesian_pe` didn't already return component masses |
| `estimate_psd_welch` | *(built, not yet wired in)* | Median-Welch PSD estimation from off-source data — implemented and numerically verified, but the orchestrator/dataset still hand the agent a ground-truth PSD directly; see Known Limitations |

`final_answer` (built-in smolagents tool) terminates each stage.

## PE design

`run_bayesian_pe` samples from a prior specification the **planning agent
builds itself** — this is not fixed by the tool author beyond validation.

- **Sampler**: dynesty nested sampling. `nlive`, `sample` method (`rwalk`/
  `slice`/`rslice`/`unif`), `walks`, `nact`, `dlogz` are all chosen per-task
  by the planning agent based on the matched-filter seed's SNR, the time
  budget, and how many dimensions it chose to free.
- **Always sampled** (never fixed — matching standard full-PE practice,
  where an uninformative posterior on a weakly-constrained parameter is the
  correct honest result, not a reason to exclude it from the model):
  `chirp_mass`, `mass_ratio`, `a_1`, `a_2`, `theta_jn`.
- **Conditionally sampled**: `tilt_1`, `tilt_2`, `phi_12`, `phi_jl` — only
  if the planning agent chooses a precession-capable approximant
  (`IMRPhenomXPHM`); rejected outright by `build_full_priors` otherwise.
- **Sky/orientation**: `ra`, `dec`, `psi` always fixed to the task's given
  values (`DeltaFunction`) — never estimated.
- **Marginalised analytically**: `geocent_time`, `luminosity_distance`,
  `phase` — not sampled as explicit dimensions, reconstructed from the
  marginalised posterior afterward.
- **Coalescence-time window**: `geocent_time`'s prior spans `±0.1s` around
  the matched-filter seed's merger-time estimate (the interferometer's time
  axis is shifted via `start_time=-merger_time_s` so `geocent_time=0`
  corresponds exactly to the seed's estimate). This exists to prevent the
  sampler from locking onto a spurious noise fluctuation elsewhere in the
  segment — see Known Limitations for current tightness.

### Approximant capabilities

Rather than a static per-approximant spin-limit dict, capabilities are
looked up via `list_approximant_capabilities()`:

```python
{
    "IMRPhenomD":    {"precession": False, "higher_modes": False, "relative_cost": "fast", ...},
    "IMRPhenomXPHM": {"precession": True,  "higher_modes": True,  "relative_cost": "slow", ...},
}
```

Only these two are currently registered for **recovery** (injection-side
generation also supports `SEOBNRv4`/`IMRPhenomXHM`, but the agent can't yet
choose them for PE — see Known Limitations). `a_1`/`a_2` use a uniform
`±0.99` bound regardless of approximant; when paired with a freed `tilt_1`/
`tilt_2` under a precessing approximant, `a_1`/`a_2` switch from a signed
aligned-spin-z convention to non-negative magnitude — `build_full_priors`
enforces this automatically.

## Agent configuration

Each stage has its own file (`agents/planning_agent_v2.py`,
`agents/execution_agent_v2.py`, `agents/critic_agent_v2.py`,
`agents/report_agent_v2.py`) and its own prompt YAML
(`prompts/planning_agent.yaml`, etc.), following this pattern:

```python
CodeAgent(
    tools=[...],                # only this stage's tools
    model=model,
    prompt_templates=prompt_templates,
    max_steps=5,
    planning_interval=None,     # critical -- see Debugging
    additional_authorized_imports=[
        "numpy", "pycbc", "json", "math",
    ],
    executor_kwargs={"timeout_seconds": 300},
)
```

## Output

The reporting agent writes `output.json` to the path specified in
`input.json`. **Seven** keys, not six — `coalescence_time_s` is required:

```json
{
    "chirp_mass_Msun":    28.04,
    "mass1_Msun":         35.47,
    "mass2_Msun":         28.75,
    "mass_ratio":         0.810,
    "network_snr":        23.09,
    "coalescence_time_s": 8.8931,
    "merger_type":        "BBH"
}
```

Only `chirp_mass_Msun` and `coalescence_time_s` are actually scored by the
benchmark evaluator — the rest are reported as diagnostic context. Plots
are saved to `GW_merger_bench/results/plots/{task_id}_chirp.png`.

## Repo structure

```
physics_agent_harness/
├── agents/
│   ├── planning_agent_v2.py    — planning stage: seed + own prior_spec + sampler config
│   ├── execution_agent_v2.py   — execution stage: run_bayesian_pe only
│   ├── critic_agent_v2.py      — critic stage: diagnostics + accept/retry decision
│   └── report_agent_v2.py      — reporting stage: classify + plot + final answer
├── prompts/
│   ├── planning_agent.yaml
│   ├── execution_agent.yaml
│   ├── critic_agent.yaml
│   └── report_agent.yaml
├── tools/
│   └── gw_tools_v2.py           — full tool set, see table above
├── run_gw_multi_v2.py           — orchestrator entry point (called by the benchmark)
├── unit_test_tools.py           — standalone tool unit tests
└── .env.example
```

## Known limitations

Worth knowing before relying on results, or before extending the pipeline:

- **`seed_pe_prior_via_matched_filter`'s coarse `mass_ratio` grid only
  spans `0.3-1.0`**, but the benchmark's hard tier can generate genuinely
  asymmetric systems down to `mass_ratio=0.1`. Hard-tier tasks with very
  unequal masses can get seeded from a grid that structurally cannot find
  the right region — this is a coverage gap, not a resolution issue.
- **`reference_frequency` is hardcoded to `20.0`** in `run_bayesian_pe`'s
  waveform arguments, independently of the `f_lower` parameter. Currently
  harmless (every task so far uses `f_lower=20.0`), but would silently
  diverge if a dataset were ever generated with a different `f_lower`.
- **`check_waveform_residual` and `plot_chirp_signal` silently default to
  `IMRPhenomD`** if the calling agent's generated code omits the
  `approximant=` argument — unlike `run_bayesian_pe`, which self-corrects
  via `priors_package`. A dropped argument here produces a
  plausible-but-wrong result rather than an error.
- **The `±0.1s` `geocent_time` window** is a real fix for a confirmed
  false-lock failure mode (the sampler occasionally converging on a random
  noise fluctuation elsewhere in the segment instead of the true merger),
  but is still roughly `30×` wider than the matched-filter seed's
  demonstrated timing accuracy — a tighter window may reduce residual risk
  further.
- **PSD is currently given directly**, not estimated by the agent from
  off-source data. `estimate_psd_welch` exists and is numerically verified,
  but isn't yet wired into `load_gw_data`/the orchestrator/the dataset
  generator — see the tool table above.
- Only `IMRPhenomD` and `IMRPhenomXPHM` are registered in
  `list_approximant_capabilities` for recovery. `SEOBNRv4`/`IMRPhenomXHM`
  work for dataset *generation* but would be rejected by `build_full_priors`
  if an agent tried to select them for recovery.

## Debugging

Common issues and fixes:

| Error | Fix |
|---|---|
| `DocstringParsingException: Cannot generate JSON schema for <fn> because it has no docstring!` | Every `@tool`-decorated function requires a docstring — smolagents uses it to build the tool's schema at import time. A missing docstring fails at import, before any agent even runs. |
| `Import of os is not allowed` | Add `"os", "pathlib"` to `additional_authorized_imports` |
| Planning fires at step 1 unexpectedly | Set `planning_interval=None`, not an integer — `(step-1) % interval == 0` fires at step 1 for any integer value |
| `FileNotFoundError` on plot output | `plot_chirp_signal` creates the output directory internally via `os.makedirs` — check `output_path`'s parent exists in the *caller's* filesystem view if this still fails |
| `ZeroDivisionError` in smolagents | `planning_interval=0` — set to `None` instead |
| `ModuleNotFoundError` after renaming an agent/tools file | Update the corresponding `from agents.X import` / `from tools.X import` line in `run_gw_multi_v2.py` and in each agent file — renaming a file doesn't update its import references anywhere else |
| Slice sampler warning from dynesty | Cosmetic only — add `warnings.filterwarnings("ignore", message="Specifying slice option")` |