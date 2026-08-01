# GW Merger Bench — Deferred Changes (for later)

## 1. V3: Genuine precession detection (not just hint-following)
**Status:** Plan only, nothing built.

- Planning agent defaults to `IMRPhenomD` regardless of hint (or still reads hint as a starting suggestion).
- Critic evaluates fit quality as now (`chi2_reduced`, `log_bayes_factor`).
- **New:** if fit looks poor *and* SNR is high enough that precession would be
  statistically detectable, critic recommends a new retry type: switch
  approximant to `IMRPhenomXPHM`, free `tilt_1/tilt_2/phi_12/phi_jl`.
- **New field:** `package_pe_critique` gains `retry_switch_precessing: bool`
  (or a full `retry_priors_package` dict).
- **`main()`'s retry loop** needs new handling — currently only mutates
  `config`; this path must rebuild `priors_package` entirely.
- **Model selection:** compare `log_bayes_factor` between non-precessing and
  precessing runs; only prefer the precessing result if evidence meaningfully
  improves (real Occam's-razor behavior, not automatic upgrade).

## 2. Harden `check_waveform_residual` / `plot_chirp_signal` against a dropped `approximant` arg
**Status:** Identified risk, not fixed.

- Both default to `"IMRPhenomD"` silently if the caller's generated code
  omits `approximant=`. Unlike `run_bayesian_pe` (self-corrects via
  `priors_package.get("approximant", approximant)`), these have no such
  fallback — a dropped argument produces a **wrong-but-plausible** result,
  not a crash.
- Fix: either remove the default (forces a loud `TypeError` if omitted) or
  thread `priors_package` itself through so the same `.get(...)` pattern
  applies.

## 3. Remove `best_mass_ratio` / `best_mass1` / `best_mass2` from the coarse seed's return
**Status:** Optional hardening, discussed, never applied.

- `seed_pe_prior_via_matched_filter` still returns these fields even though
  we proved the coarse search's mass-ratio point estimate is fundamentally
  unreliable (genuine near-degeneracy, confirmed via `diagnose_q_landscape.py`).
- Not required — the wide `mass_ratio` prior default already protects
  `chirp_mass` accuracy — but removing these fields would make it
  *structurally impossible* for an agent to build a narrow window around an
  untrustworthy number, rather than merely discouraged by prompt text.
- Keep the internal grid search using `q` (needed for accurate `best_snr`);
  just stop **returning** the three derived fields.

## 4. Register `SEOBNRv4` and `IMRPhenomXHM` in `list_approximant_capabilities`
**Status:** Offered, not applied.

- Both already work for **injection** (`generate_dataset.py --approximant`);
  neither is recognized for **recovery** — `build_full_priors` would reject
  them today.
- Add entries:
  - `SEOBNRv4`: aligned-spin, EOB formalism (different theoretical approach
    than IMRPhenom family) — useful for waveform-systematics testing.
  - `IMRPhenomXHM`: aligned-spin, adds higher-order harmonic modes — matters
    for asymmetric mass ratio / inclined systems.

## 5. `generate_dataset.py` — occasional hard-tier generation failures under precession
**Status:** Observed, not fixed.

- Task generation can exhaust all `max_attempts` (20) and fail entirely,
  more often on `hard` tier with `--enable-precession` — precessing signals'
  amplitude/phase modulation makes the Hilbert-transform freq-evo
  measurement noisier, failing the 10% acceptance tolerance more often.
- No retry/resume mechanism exists for a single failed task index (unlike
  `run_benchmark.py`'s `--start-from-task-id`) — a failed slot is just
  skipped, dataset ends up short.
- Possible fixes: raise `max_attempts` for hard tier under precession,
  loosen the freq-evo tolerance, or add per-task regeneration support.

## 6. Glitch/calibration *recovery*-side tooling (from the original realism roadmap)
**Status:** Injection built; agent-side detection/handling never built.

- Glitch **injection** exists (`_inject_glitch`); no `detect_glitch_candidates`
  or `gate_strain_segment` tool exists for the agent to actually notice or
  handle a glitch in the data.
- Calibration-error **injection** exists (`_apply_calibration_distortion`,
  tier-scaled); no corresponding **recovery**-side modeling exists (e.g. a
  `CalibrationPriorDict`-style spline-node extension to `run_bayesian_pe`).

## 7. `geocent_time` prior window — confirmed bug, fix identified, not yet applied
**Status:** Root-caused (task 003, easy tier, post-noise-fix), fix specified, not applied to the real file.

- `run_bayesian_pe`'s `geocent_time` prior currently spans the *entire*
  segment (`±duration/2`, i.e. `±8s`) rather than a tight window around the
  matched-filter seed's `merger_time_s` — which the seed has consistently
  measured to `~0.003s` accuracy (`diagnose_time_bug.py`).
- Now that noise has its correct, real amplitude (post the `_colored_noise`
  fix), a random noise fluctuation elsewhere in the segment can occasionally
  present a locally-strong match, and the sampler can lock onto it instead
  of the true merger — confirmed happening on task 003 (recovered
  `coalescence_time_s=0.92` vs. true `8.89`, with mass/spin fit to the wrong
  part of the data as a consequence).
- Fix: narrow the prior to `±0.02s` (a `~6-7x` margin over the seed's
  demonstrated accuracy, not an arbitrary generic "rapid-PE" number):
  ```python
  TIME_WINDOW_S = 0.02
  priors["geocent_time"] = bilby.core.prior.Uniform(
      minimum=-TIME_WINDOW_S, maximum=TIME_WINDOW_S, name="geocent_time"
  )
  ```
  (The frame is already shifted via `start_time=-peak_time` so `0` in this
  window means "exactly at the seed's `merger_time_s`" — no additional
  anchoring code needed, just the tighter bound.)
- Needs applying to the real `run_bayesian_pe` before further benchmark runs
  are trustworthy — any run made before this fix (including the full
  zero-spin/aligned-spin/precessing benchmark runs planned right after the
  noise-calibration fix) is at risk of this same false-lock failure mode.

## 8. Full off-source PSD-estimation feature — explicitly deferred, mid-design
**Status:** Tool built and verified (`estimate_psd_welch`, confirmed correct via
`verify_psd_estimation.py`); NOT yet wired into the real pipeline. Deliberately
postponed by design: agreed to first test the original tool-flow pipeline
against the noise-recalibrated datasets, before adding this on top.

Remaining work, all still needed:
- `generate_dataset.py`: save a separate off-source (signal-free) noise
  segment per task (`OFFSOURCE_DURATION`, e.g. 128s), independent noise seed
  from on-source; add `strain_H1_offsource`/`strain_L1_offsource` to
  `data_files`; move `psd_H1`/`psd_L1`/`psd_freqs` out of agent-visible
  `data_files` into ground-truth-only `true_psd_*.npy` files.
- `load_gw_data`: load off-source strain instead of a ready-made PSD.
- `run_gw_multi.py`: new planning step calling `estimate_psd_welch` on both
  detectors' off-source data before the matched-filter seed step; thread the
  resulting estimated PSD (not any ground truth) through every subsequent
  stage's `additional_args`.
- `planning_agent.yaml` / task descriptions: explain that PSD is no longer
  given, must be estimated first.
- No changes needed to `seed_pe_prior_via_matched_filter`, `run_bayesian_pe`,
  `check_waveform_residual`, `plot_chirp_signal` themselves — all already
  PSD-agnostic, just need the estimated array passed in instead of the given one.

## 9. Expose currently-hardcoded tool parameters as agent decisions
**Status:** Idea only, for a future "no fixed tool-flow" test mode — testing
whether the agent can reason well about methodological/numeric knobs
currently fixed by the tool author, not just the physics priors it already
controls. Would replace prescriptive step-by-step tool-call prompts with a
more open decision space. Extended after a full line-by-line pass through
the real `gw_tools_v2.py`.

**`run_bayesian_pe`**:
- `geocent_time` window width (`TIME_WINDOW_S`, currently `0.1`) — could
  instead be an agent-chosen multiple of the seed's own reported timing
  precision, rather than one fixed constant for every task regardless of SNR.
- `luminosity_distance` prior: bounds (`10-5000 Mpc`) and `alpha=2` power-law
  index — all fixed.
- `phase` prior (`Uniform(0, 2π)`) — arguably never needs to change, lowest
  priority of this group.
- Default `chirp_mass` window multipliers (`Mc_guess*0.50` to `Mc_guess*1.70`)
  and absolute floor/ceiling (`2.0`/`150.0`) used only when the agent omits
  `chirp_mass` from `prior_spec`.
- PSD safety-floor substitution value (`1e-38`) for invalid/zero PSD bins.
- `reference_frequency=20.0` — see item 10 below, this one may be a live bug
  more than a "future choice," since it's currently decoupled from `f_lower`.
- `n_cores = os.cpu_count() - 2` — a resource-allocation heuristic ("leave 2
  cores free"), not physics, but still a currently-fixed choice.

**`seed_pe_prior_via_matched_filter`**:
- Coarse grid bounds/resolution: `chirp_masses` (`4-90 Msun`, 30 points log-
  spaced), `mass_ratios` (`0.3-1.0`, 30 points) — see item 10, the `q` floor
  is a live coverage bug, not just a future-choice item.
- Stage-2 refinement window width (±1 coarse grid step) and resolution
  (15 fine points) — could be sized based on time budget.
- `inverse_spectrum_truncation` window length (`4 * sample_rate`, i.e. 4s) —
  a PSD-conditioning parameter, currently fixed.

**`check_waveform_residual`**:
- The ±10ms / 41-step local time-search window for the sub-ms timing
  correction — currently a fixed constant, unrelated to the actual timing
  precision (`coalescence_time_ci_width`) of whatever PE run produced the
  inputs; could scale with that instead of being one-size-fits-all.
- `n_bins` is already an exposed parameter (default 8) but never actually
  varied by any agent so far.

**`package_pe_critique`**:
- The `chi2_reduced > 3.0` / `log_bayes_factor < 0` acceptance thresholds
  are hardcoded validation constants — could instead be argued for/against
  by the critic given SNR, dimensionality, or dataset difficulty.

**`estimate_psd_welch`** (once wired in per item 8):
- `seg_duration`/`overlap_frac` are already tool arguments with defaults,
  but no prompt currently asks the agent to reason about or justify a
  non-default choice.

**`plot_chirp_signal`** (diagnostic-only, lowest priority — doesn't affect
scored results):
- Whitening edge-trim (`0.5s` from each end), Q-transform display window
  (`±0.5s` around merger), `qrange=(4, 64)`, upper display frequency (`512Hz`).

Physical floors/ceilings in `PARAM_SCHEMA` and the `nlive≥20` safety minimum
in `build_pe_config` should probably stay hardcoded regardless — those are
genuine physical/numerical-stability limits, not methodological choices.

## 10. Two genuine live issues found during the item-9 audit (not just future-choice candidates)
**Status:** Confirmed via direct code inspection, not yet fixed.

- **`reference_frequency=20.0` is hardcoded independently of `f_lower`** in
  `run_bayesian_pe`'s `waveform_arguments`. Every task so far has had
  `f_lower=20.0` so this has never actually diverged, but it's structurally
  fragile — should be derived from (or tied to) the `f_lower` parameter
  rather than a separate literal, especially since `reference_frequency`
  also affects the precession-angle conversion
  (`bilby_to_lalsimulation_spins`) elsewhere in the pipeline.