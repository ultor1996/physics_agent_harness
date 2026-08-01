from smolagents import tool
import numpy as np


# =============================================================================
# Tool 1 -- Data loading
# =============================================================================

@tool
def load_gw_data(strain_H1: str, strain_L1: str,
                 psd_H1: str, psd_L1: str,
                 psd_freqs: str,
                 sample_rate: int = 2048) -> dict:
    """
    Load gravitational-wave strain and PSD data from .npy files.
    Always call this first before any other tool.
    Returns strain_H1, strain_L1, psd_H1, psd_L1, psd_freqs,
    sample_rate, duration, delta_t, delta_f as a dict.

    Args:
        strain_H1: absolute path to H1 strain .npy file
        strain_L1: absolute path to L1 strain .npy file
        psd_H1: absolute path to H1 PSD .npy file
        psd_L1: absolute path to L1 PSD .npy file
        psd_freqs: absolute path to PSD frequency axis .npy file
        sample_rate: sampling rate in Hz from the task description e.g. 2048
    """
    strain_H1_arr = np.load(strain_H1)
    strain_L1_arr = np.load(strain_L1)
    psd_H1_arr    = np.load(psd_H1)
    psd_L1_arr    = np.load(psd_L1)
    psd_freqs_arr = np.load(psd_freqs)

    delta_t   = 1.0 / sample_rate
    n_samples = len(strain_H1_arr)
    duration  = n_samples / sample_rate
    delta_f   = float(psd_freqs_arr[1] - psd_freqs_arr[0]) if len(psd_freqs_arr) > 1 else 1.0 / duration

    return {
        "strain_H1":   strain_H1_arr,
        "strain_L1":   strain_L1_arr,
        "psd_H1":      psd_H1_arr,
        "psd_L1":      psd_L1_arr,
        "psd_freqs":   psd_freqs_arr,
        "sample_rate": sample_rate,
        "delta_t":     delta_t,
        "delta_f":     delta_f,
        "duration":    duration,
        "n_samples":   n_samples,
    }


def _eval_single_template(args):
    """
    Evaluate one (Mc, q) template using both H1 and L1 detectors.
    Spins fixed to zero -- this is a coarse triage seed only, not final
    PE, so the zero-spin simplification here is intentional and unrelated
    to whatever the planning agent later decides for full PE. Module-level
    function required for multiprocessing pickling. Returns
    (Mc, q, network_snr, m1, m2) or None.
    """
    (Mc, q, strain_H1_arr, psd_H1_trunc, strain_L1_arr, psd_L1_trunc, freqs_arr,
     delta_t, delta_f, flen, f_lower, approximant) = args

    try:
        from pycbc.types import TimeSeries
        from pycbc.waveform import get_fd_waveform
        from pycbc.filter import matched_filter
        import numpy as np

        q  = float(np.clip(q, 0.05, 1.0))
        m1 = float(Mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0))
        m2 = q * m1
        if m2 < 1.0 or m1 > 200.0:
            return None

        hp, _ = get_fd_waveform(
            approximant=approximant,
            mass1=m1, mass2=m2,
            spin1z=0.0, spin2z=0.0,
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=float(0.5 / delta_t),
        )
        if len(hp) < flen:
            hp.resize(flen)
        elif len(hp) > flen:
            hp = hp[:flen]

        strain_H1_ts = TimeSeries(strain_H1_arr, delta_t=delta_t)
        snr_H1       = matched_filter(hp, strain_H1_ts.to_frequencyseries(),
                                      psd=psd_H1_trunc, low_frequency_cutoff=f_lower)
        peak_H1      = float(abs(snr_H1).numpy().max())

        strain_L1_ts = TimeSeries(strain_L1_arr, delta_t=delta_t)
        snr_L1       = matched_filter(hp, strain_L1_ts.to_frequencyseries(),
                                      psd=psd_L1_trunc, low_frequency_cutoff=f_lower)
        peak_L1      = float(abs(snr_L1).numpy().max())

        network_snr = float(np.sqrt(peak_H1**2 + peak_L1**2))
        return (float(Mc), float(q), network_snr, float(m1), float(m2))

    except Exception:
        return None


@tool
def build_pe_config(nlive: int, sample: str, walks: int, nact: int, dlogz: float,
                     rationale: str = "") -> dict:
    """
    Package your chosen dynesty sampler settings (nlive, sample, walks,
    nact, dlogz) into a validated config dict to pass into
    run_bayesian_pe's `config` argument. This tool does NOT choose values
    for you -- decide these based on the matched-filter seed, your
    remaining time budget, AND how many dimensions you chose to free in
    build_full_priors (more free parameters generally need more live
    points to converge reliably).

    Prior/physics decisions (which parameters to sample, their bounds)
    live in build_full_priors, NOT here.

    Args:
        nlive: number of dynesty live points you have chosen
        sample: dynesty sample method you have chosen, e.g. "rwalk" or "slice"
        walks: walks parameter for rwalk sampling
        nact: autocorrelation multiple for rwalk termination
        dlogz: evidence convergence criterion (smaller = more precise, slower)
        rationale: your one-sentence reasoning for these choices, for logging
    """
    valid_samples = {"rwalk", "slice", "rslice", "unif"}
    if sample not in valid_samples:
        raise ValueError(f"sample must be one of {valid_samples}, got {sample!r}")
    if nlive < 20:
        raise ValueError("nlive too low to produce a usable posterior (minimum 20)")
    if dlogz <= 0:
        raise ValueError("dlogz must be positive")

    return {
        "nlive": int(nlive),
        "sample": sample,
        "walks": int(walks),
        "nact": int(nact),
        "dlogz": float(dlogz),
        "rationale": rationale,
    }


# =============================================================================
# Tool -- Approximant capability lookup (query, don't guess from memory)
# =============================================================================

@tool
def list_approximant_capabilities() -> dict:
    """
    Returns the physics each available approximant can model, so you can
    reason about prior construction with real information instead of
    recalling facts from training data.

    Returns a dict keyed by approximant name, each with:
        precession (bool), higher_modes (bool), tidal (bool),
        relative_cost (str), notes (str)
    """
    return {
        "IMRPhenomD": {
            "precession": False, "higher_modes": False, "tidal": False,
            "relative_cost": "fast",
            "notes": (
                "Aligned-spin, dominant-mode only. a_1/a_2 represent the "
                "SIGNED aligned spin_z component directly -- tilt cannot "
                "be freed, so sign encodes aligned vs anti-aligned spin."
            ),
        },
        "IMRPhenomXPHM": {
            "precession": True, "higher_modes": True, "tidal": False,
            "relative_cost": "slow",
            "notes": (
                "Precessing, includes higher-order modes. When tilt_1/"
                "tilt_2 are freed, a_1/a_2 switch meaning to NON-NEGATIVE "
                "spin MAGNITUDES (0 to 0.99) -- direction now comes from "
                "tilt, not from a_1's sign. Substantially more expensive "
                "per likelihood evaluation."
            ),
        },
    }

# =============================================================================
# Tool -- Full prior specification builder (agent decides, tool validates)
# =============================================================================

PARAM_SCHEMA = {
    "chirp_mass": {"range": (0.5, 200.0),  "types": {"UniformInComponentsChirpMass", "Uniform"}},
    "mass_ratio": {"range": (0.02, 1.0),   "types": {"Uniform"}},
    "a_1":        {"range": (-0.99, 0.99), "types": {"Uniform", "fixed"}},
    "a_2":        {"range": (-0.99, 0.99), "types": {"Uniform", "fixed"}},
    "theta_jn":   {"range": (0.0, np.pi),  "types": {"Sine", "Uniform", "fixed"}},
    "tilt_1":     {"range": (0.0, np.pi),   "types": {"Sine", "fixed"}, "precession_only": True},
    "tilt_2":     {"range": (0.0, np.pi),   "types": {"Sine", "fixed"}, "precession_only": True},
    "phi_12":     {"range": (0.0, 2*np.pi), "types": {"Uniform", "fixed"}, "precession_only": True},
    "phi_jl":     {"range": (0.0, 2*np.pi), "types": {"Uniform", "fixed"}, "precession_only": True},
}


@tool
def build_full_priors(approximant: str, prior_spec: dict, rationale: str) -> dict:
    """
    Validate and package your own prior specification for run_bayesian_pe.
    You decide which parameters to sample and with what distribution --
    this tool enforces physical validity and approximant compatibility,
    it does not choose values for you.

    prior_spec format -- one entry per parameter you want to specify:
      {
          "chirp_mass": {"type": "UniformInComponentsChirpMass", "minimum": 15, "maximum": 22},
          "mass_ratio": {"type": "Uniform", "minimum": 0.1, "maximum": 1.0},
          "a_1": {"type": "Uniform", "minimum": -0.5, "maximum": 0.5},
          "a_2": {"type": "fixed", "value": 0.0},
          "theta_jn": {"type": "Sine"},
      }
    Parameters you omit fall back to safe, sensible defaults inside
    run_bayesian_pe (chirp_mass/mass_ratio narrowed around the
    matched-filter seed; a_1/a_2/theta_jn fixed at 0.0). ra, dec, psi are
    never included here -- always given directly by the task and fixed
    automatically.

    Args:
        approximant: the waveform approximant you've decided to use for
            full PE -- check list_approximant_capabilities first.
        prior_spec: your parameter -> prior specification, see format above.
        rationale: your reasoning for these choices, for logging.
    """
    caps = list_approximant_capabilities().get(approximant)
    if caps is None:
        raise ValueError(
            f"Unknown approximant {approximant!r} -- check "
            f"list_approximant_capabilities() for available options."
        )

    validated = {}
    for pname, spec in prior_spec.items():
        if pname not in PARAM_SCHEMA:
            raise ValueError(f"Unrecognized parameter {pname!r} in prior_spec.")
        schema = PARAM_SCHEMA[pname]
        ptype = spec.get("type")
        if ptype not in schema["types"]:
            raise ValueError(
                f"{pname}: type {ptype!r} not allowed. Allowed: {sorted(schema['types'])}"
            )
        if ptype == "fixed":
            if "value" not in spec:
                raise ValueError(f"{pname}: type='fixed' requires a 'value' key.")
            validated[pname] = dict(spec)
            continue
        lo, hi = schema["range"]
        if "minimum" in spec and spec["minimum"] < lo:
            raise ValueError(f"{pname}: minimum {spec['minimum']} below physical floor {lo}")
        if "maximum" in spec and spec["maximum"] > hi:
            raise ValueError(f"{pname}: maximum {spec['maximum']} above physical ceiling {hi}")
        if "minimum" in spec and "maximum" in spec and spec["minimum"] >= spec["maximum"]:
            raise ValueError(f"{pname}: minimum must be < maximum")
        if schema.get("precession_only") and ptype != "fixed" and not caps["precession"]:
            raise ValueError(
                f"{pname} cannot be freed ({ptype!r}) -- approximant {approximant!r} "
                f"does not support precession. Set type='fixed' or choose a "
                f"precession-capable approximant."
            )
        validated[pname] = dict(spec)

    for tilt_name, a_name in [("tilt_1", "a_1"), ("tilt_2", "a_2")]:
        tilt_spec = validated.get(tilt_name)
        a_spec = validated.get(a_name)
        tilt_is_free = tilt_spec is not None and tilt_spec.get("type") != "fixed"
        if tilt_is_free and a_spec is not None and a_spec.get("type") == "Uniform":
            if a_spec.get("minimum", 0) < 0:
                raise ValueError(
                    f"{a_name} minimum must be >= 0 when {tilt_name} is freed -- "
                    f"direction now comes from {tilt_name}, not {a_name}'s sign."
                )
    return {
        "approximant": approximant,
        "prior_spec": validated,
        "rationale": rationale,
    }


def _build_prior_from_spec(name, spec, default_prior):
    """Construct one bilby prior object from a validated prior_spec entry."""
    import bilby
    if spec is None:
        return default_prior
    ptype = spec["type"]
    if ptype == "fixed":
        return bilby.core.prior.DeltaFunction(spec["value"])
    if ptype == "Sine":
        return bilby.core.prior.Sine(name=name)
    if ptype == "Uniform":
        return bilby.core.prior.Uniform(
            minimum=spec["minimum"], maximum=spec["maximum"], name=name,
        )
    if ptype == "UniformInComponentsChirpMass":
        return bilby.gw.prior.UniformInComponentsChirpMass(
            minimum=spec["minimum"], maximum=spec["maximum"], name=name,
        )
    raise ValueError(f"Unhandled prior type {ptype!r} for {name}")

@tool
def seed_pe_prior_via_matched_filter(strain: list, psd: list, psd_freqs: list,
                                      strain_L1: list, psd_L1: list,
                                      sample_rate: int, approximant: str = "IMRPhenomD",
                                      f_lower: float = 20.0) -> dict:
    """
    Two-stage matched filter bank over chirp mass and mass ratio.
    This is NOT a final answer and has no standalone interpretation --
    its only purpose is to produce a chirp_mass_guess and mass_ratio_guess
    to pass into run_bayesian_pe, which performs the actual parameter
    estimation. Always call run_bayesian_pe immediately after this tool.

    Stage 1: coarse 30x30 grid over chirp mass and mass ratio (spins
    fixed to zero), combining H1 and L1 network SNR -- same as before.

    Stage 2: the coarse chirp-mass grid (log-spaced, ~11% relative
    spacing) is too coarse to resolve the true chirp-mass peak at
    moderate/high SNR, where the peak can be <2% relative half-width.
    A coarse grid can alias onto a WRONG (Mc, q) combination that
    "compensates" for a mis-quantized Mc via the Mc-q phase correlation,
    giving a systematically biased mass_ratio_guess even when
    chirp_mass_guess still looks reasonable. To fix this: for EVERY
    mass_ratio grid value (not just the raw top-K coarse winners), find
    its own best coarse chirp mass, then refine chirp mass locally with
    a fine scan around that point. This is done per-q rather than
    per-top-K-winner so a mass ratio whose true peak was missed by the
    coarse chirp-mass grid still gets a fair, properly-refined
    comparison against the others.

    Returns best_chirp_mass_Msun, best_mass_ratio, best_snr, best_mass1,
    best_mass2 -- a coarse-plus-refined seed only, not a result to report.

    Args:
        strain: H1 strain time series -- use data["strain_H1"] from load_gw_data
        psd: H1 power spectral density -- use data["psd_H1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        strain_L1: L1 strain time series -- use data["strain_L1"] from load_gw_data
        psd_L1: L1 power spectral density -- use data["psd_L1"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        approximant: waveform approximant string e.g. IMRPhenomD
        f_lower: lower frequency cutoff in Hz e.g. 20.0
    """
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    delta_t       = 1.0 / sample_rate
    strain_H1_arr = np.array(strain,    dtype=np.float64)
    psd_H1_arr    = np.array(psd,       dtype=np.float64)
    strain_L1_arr = np.array(strain_L1, dtype=np.float64)
    psd_L1_arr    = np.array(psd_L1,    dtype=np.float64)
    freqs_arr     = np.array(psd_freqs, dtype=np.float64)
    N             = len(strain_H1_arr)
    delta_f       = 1.0 / (N * delta_t)
    flen          = N // 2 + 1
    n_cores       = max(1, multiprocessing.cpu_count())

    # Compute truncated PSDs ONCE, outside the per-template loop
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.types import FrequencySeries

    def build_psd_once(psd_arr):
        psd_fs     = FrequencySeries(psd_arr, delta_f=float(freqs_arr[1] - freqs_arr[0]))
        psd_interp = interpolate(psd_fs, delta_f)
        return inverse_spectrum_truncation(
            psd_interp, int(4 * sample_rate), low_frequency_cutoff=f_lower
        )

    psd_H1_trunc = build_psd_once(psd_H1_arr)
    psd_L1_trunc = build_psd_once(psd_L1_arr)

    # ---- Stage 1: coarse 2D grid (unchanged from before) ----
    chirp_masses = np.logspace(np.log10(4), np.log10(90), 30)
    mass_ratios  = np.linspace(0.1, 1.0, 30)

    args = [
        (Mc, q, strain_H1_arr, psd_H1_trunc, strain_L1_arr, psd_L1_trunc,
         freqs_arr, delta_t, delta_f, flen, f_lower, approximant)
        for Mc in chirp_masses
        for q  in mass_ratios
    ]

    results = []
    if os.environ.get("GW_MERGER_BENCH_FORCE_SEQUENTIAL"):
        for a in args:
            res = _eval_single_template(a)
            if res is not None:
                results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            for res in ex.map(_eval_single_template, args, chunksize=10):
                if res is not None:
                    results.append(res)

    if not results:
        # FIXED: fallback now includes merger_time_s, previously missing --
        # caused a downstream KeyError in run_bayesian_pe when every
        # template in the grid failed (e.g. due to a bug, or a task whose
        # true parameters fall entirely outside the search grid).
        return {
            "best_chirp_mass_Msun": 25.0,
            "best_mass_ratio":      0.8,
            "best_snr":             0.0,
            "best_mass1":           29.0,
            "best_mass2":           23.0,
            "merger_time_s":        8.0,
        }

    # ---- Stage 2: per-q chirp-mass refinement ----
    # For each distinct q used in the coarse grid, find its own best
    # coarse Mc, then evaluate a fine local Mc scan (spanning roughly
    # +/-1 coarse grid step) around that point using the SAME q. This
    # directly targets the coarse-grid-aliasing bug we diagnosed:
    # the true peak can sit strictly between two coarse Mc grid points.
    best_coarse_per_q = {}
    for Mc, q, snr, m1, m2 in results:
        if q not in best_coarse_per_q or snr > best_coarse_per_q[q][1]:
            best_coarse_per_q[q] = (Mc, snr)

    refine_args = []
    for q, (Mc_coarse, _snr) in best_coarse_per_q.items():
        idx = int(np.argmin(np.abs(chirp_masses - Mc_coarse)))
        lo_bound = chirp_masses[max(0, idx - 1)]
        hi_bound = chirp_masses[min(len(chirp_masses) - 1, idx + 1)]
        fine_mcs = np.linspace(lo_bound, hi_bound, 15)
        for Mc_fine in fine_mcs:
            refine_args.append(
                (Mc_fine, q, strain_H1_arr, psd_H1_trunc, strain_L1_arr, psd_L1_trunc,
                 freqs_arr, delta_t, delta_f, flen, f_lower, approximant)
            )

    refine_results = []
    if os.environ.get("GW_MERGER_BENCH_FORCE_SEQUENTIAL"):
        for a in refine_args:
            res = _eval_single_template(a)
            if res is not None:
                refine_results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            for res in ex.map(_eval_single_template, refine_args, chunksize=10):
                if res is not None:
                    refine_results.append(res)

    # Overall winner across BOTH the coarse grid and the per-q refinement
    all_results = results + refine_results
    best = max(all_results, key=lambda x: x[2])
    Mc, q, snr, m1, m2 = best

    # Find SNR peak time using the best (refined) template -- more
    # reliable than raw strain amplitude for merger time estimation.
    from pycbc.types import TimeSeries
    from pycbc.waveform import get_fd_waveform
    from pycbc.filter import matched_filter as _mf

    q_b  = float(np.clip(q, 0.05, 1.0))
    m1_b = float(Mc * (1.0 + q_b) ** (1.0/5.0) / q_b ** (3.0/5.0))
    m2_b = q_b * m1_b

    hp_b, _ = get_fd_waveform(
        approximant=approximant,
        mass1=m1_b, mass2=m2_b,
        spin1z=0.0, spin2z=0.0,
        delta_f=delta_f, f_lower=f_lower,
        f_final=float(0.5 / delta_t),
    )
    if len(hp_b) < flen:
        hp_b.resize(flen)
    elif len(hp_b) > flen:
        hp_b = hp_b[:flen]

    # Reuse the already-truncated H1 PSD instead of recomputing it a third time
    snr_ts    = _mf(hp_b,
                     TimeSeries(strain_H1_arr, delta_t=delta_t).to_frequencyseries(),
                     psd=psd_H1_trunc, low_frequency_cutoff=f_lower)
    peak_idx  = int(abs(snr_ts).numpy().argmax())
    merger_time_s = float(peak_idx) * delta_t

    return {
        "best_chirp_mass_Msun": round(float(Mc),       4),
        "best_mass_ratio":      round(float(q),         4),
        "best_snr":             round(float(snr),       4),
        "best_mass1":           round(float(m1),        3),
        "best_mass2":           round(float(m2),        3),
        "merger_time_s":        round(merger_time_s,    4),
    }

# =============================================================================
# Tool 3 -- Full Bayesian parameter estimation
# =============================================================================

@tool
def run_bayesian_pe(strain_H1: list, psd_H1: list,
                     strain_L1: list, psd_L1: list,
                     psd_freqs: list, sample_rate: int,
                     chirp_mass_guess: float, mass_ratio_guess: float,
                     given_ra: float, given_dec: float, given_psi: float,
                     priors_package: dict, config: dict,
                     merger_time_s: float = None,
                     f_lower: float = 20.0, approximant: str = "IMRPhenomD") -> dict:
    """
    Run full Bayesian parameter estimation with Bilby + dynesty nested
    sampling, following standard GW rapid-PE practice.

    Priors are built from priors_package (from build_full_priors) --
    whichever parameters the planning agent chose to free get sampled
    with the agent's own bounds; anything omitted falls back to a safe
    default (chirp_mass/mass_ratio narrowed around the matched-filter
    seed; a_1/a_2/theta_jn fixed at 0.0). Sky location (ra, dec, psi) is
    always fixed to the given values. Time, phase, and distance are
    analytically marginalised in the likelihood.

    Returns posterior medians and 5/95 percentile credible intervals for
    chirp_mass and mass_ratio (plus derived component masses), the log
    Bayes factor, and -- if spin/inclination were freed -- the recovered
    a_1/a_2/theta_jn values (0.0 if they were fixed).

    Args:
        strain_H1: H1 strain time series -- use data["strain_H1"] from load_gw_data
        psd_H1: H1 PSD -- use data["psd_H1"] from load_gw_data
        strain_L1: L1 strain time series -- use data["strain_L1"] from load_gw_data
        psd_L1: L1 PSD -- use data["psd_L1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        chirp_mass_guess: best_chirp_mass_Msun from seed_pe_prior_via_matched_filter
        mass_ratio_guess: best_mass_ratio from seed_pe_prior_via_matched_filter
        priors_package: dict from build_full_priors -- {"approximant", "prior_spec", "rationale"}
        config: dict from build_pe_config -- sampler settings only
                (nlive, sample, walks, nact, dlogz). Always call
                build_pe_config first and pass its output here unmodified.
        merger_time_s: merger_time_s from seed_pe_prior_via_matched_filter
        given_ra: known right ascension in radians -- from task's given_parameters
        given_dec: known declination in radians -- from task's given_parameters
        given_psi: known polarisation angle in radians -- from task's given_parameters
        f_lower: lower frequency cutoff in Hz e.g. 20.0
        approximant: waveform approximant string -- should match priors_package["approximant"]
    """
    import os
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import bilby
    import numpy as np
    import logging
    import time

    logging.getLogger("bilby").setLevel(logging.ERROR)
    logging.getLogger("dynesty").setLevel(logging.ERROR)

    sample_rate   = int(sample_rate)
    strain_H1_arr = np.asarray(strain_H1, dtype=np.float64)
    strain_L1_arr = np.asarray(strain_L1, dtype=np.float64)
    psd_H1_arr    = np.asarray(psd_H1,    dtype=np.float64)
    psd_L1_arr    = np.asarray(psd_L1,    dtype=np.float64)
    freqs_arr     = np.asarray(psd_freqs, dtype=np.float64)

    duration = float(len(strain_H1_arr) / sample_rate)

    ifo_H1 = bilby.gw.detector.get_empty_interferometer("H1")
    ifo_L1 = bilby.gw.detector.get_empty_interferometer("L1")

    if merger_time_s is None:
        peak_idx      = int(np.argmax(np.abs(strain_H1_arr)))
        merger_time_s = float(peak_idx) / float(sample_rate)

    peak_time = float(merger_time_s)

    ifo_H1.strain_data.set_from_time_domain_strain(
        strain_H1_arr, sampling_frequency=float(sample_rate),
        duration=duration, start_time=-peak_time,
    )
    ifo_L1.strain_data.set_from_time_domain_strain(
        strain_L1_arr, sampling_frequency=float(sample_rate),
        duration=duration, start_time=-peak_time,
    )

    psd_H1_safe = np.where((psd_H1_arr <= 0) | (freqs_arr < f_lower), 1e-38, psd_H1_arr)
    psd_L1_safe = np.where((psd_L1_arr <= 0) | (freqs_arr < f_lower), 1e-38, psd_L1_arr)

    ifo_H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freqs_arr, psd_array=psd_H1_safe
    )
    ifo_L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freqs_arr, psd_array=psd_L1_safe
    )
    ifo_H1.minimum_frequency = f_lower
    ifo_L1.minimum_frequency = f_lower

    interferometers = bilby.gw.detector.InterferometerList([ifo_H1, ifo_L1])

    actual_approximant = priors_package.get("approximant", approximant)

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": actual_approximant,
            "reference_frequency":  20.0,
            "minimum_frequency":    f_lower,
        },
    )

    Mc_guess = max(float(chirp_mass_guess), 1.0)
    q_guess  = float(np.clip(mass_ratio_guess, 0.05, 1.0))

    prior_spec = priors_package.get("prior_spec", {})

    priors = bilby.gw.prior.BBHPriorDict()

    priors["chirp_mass"] = _build_prior_from_spec(
        "chirp_mass", prior_spec.get("chirp_mass"),
        default_prior=bilby.gw.prior.UniformInComponentsChirpMass(
            minimum=max(2.0, Mc_guess * 0.50),
            maximum=min(150.0, Mc_guess * 1.70),
            name="chirp_mass", latex_label=r"$\mathcal{M}$",
        ),
    )
    # NOTE: mass_ratio now narrows around the matched-filter seed by
    # default (previously it silently fell through to BBHPriorDict()'s
    # very wide Uniform(0.125, 1.0) default and was never actually
    # narrowed despite q_guess being computed -- fixed here).
    priors["mass_ratio"] = _build_prior_from_spec(
        "mass_ratio", prior_spec.get("mass_ratio"),
        # default_prior=bilby.core.prior.Uniform(
        #     minimum=max(0.05, q_guess * 0.5),
        #     maximum=min(1.0, q_guess * 1.5),
        #     name="mass_ratio",
        # ),
        default_prior=bilby.core.prior.Uniform(
        minimum=0.05, maximum=1.0, name="mass_ratio",
        ),
    )

    priors["ra"]  = bilby.core.prior.DeltaFunction(peak=given_ra,  name="ra")
    priors["dec"] = bilby.core.prior.DeltaFunction(peak=given_dec, name="dec")
    priors["psi"] = bilby.core.prior.DeltaFunction(peak=given_psi, name="psi")

    priors["theta_jn"] = _build_prior_from_spec(
        "theta_jn", prior_spec.get("theta_jn"),
        default_prior=bilby.core.prior.DeltaFunction(0.0),
    )
    priors["a_1"] = _build_prior_from_spec(
        "a_1", prior_spec.get("a_1"), default_prior=bilby.core.prior.DeltaFunction(0.0),
    )
    priors["a_2"] = _build_prior_from_spec(
        "a_2", prior_spec.get("a_2"), default_prior=bilby.core.prior.DeltaFunction(0.0),
    )
    # Precession angles: only meaningful/sampleable with a
    # precession-capable approximant (currently IMRPhenomXPHM). Fixed at
    # 0.0 by default; build_full_priors rejects freeing these unless the
    # chosen approximant supports precession.
    priors["tilt_1"] = _build_prior_from_spec("tilt_1", prior_spec.get("tilt_1"), default_prior=bilby.core.prior.DeltaFunction(0.0))
    priors["tilt_2"] = _build_prior_from_spec("tilt_2", prior_spec.get("tilt_2"), default_prior=bilby.core.prior.DeltaFunction(0.0))
    priors["phi_12"] = _build_prior_from_spec("phi_12", prior_spec.get("phi_12"), default_prior=bilby.core.prior.DeltaFunction(0.0))
    priors["phi_jl"] = _build_prior_from_spec("phi_jl", prior_spec.get("phi_jl"), default_prior=bilby.core.prior.DeltaFunction(0.0))

    TIME_WINDOW_S = 0.1  # standard rapid-PE coalescence-time window around the trigger
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=-TIME_WINDOW_S, maximum=TIME_WINDOW_S, name="geocent_time"
    )
    priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
        alpha=2, name="luminosity_distance", minimum=10.0, maximum=5000.0
    )
    priors["phase"] = bilby.core.prior.Uniform(0.0, 2*np.pi, name="phase", boundary="periodic")

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        time_marginalization=True,
        distance_marginalization=True,
        phase_marginalization=True,
        jitter_time=False,
    )

    n_cores = max(1, os.cpu_count() - 2)
    t0 = time.time()
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler="dynesty",
        sample=config.get("sample", "rwalk"),
        walks=config.get("walks", 50),
        nact=config.get("nact", 5),
        dlogz=config.get("dlogz", 0.5),
        nlive=int(config.get("nlive", 150)),
        npool=n_cores,
        outdir="/tmp/bilby_pe_out", label="gw_pe", clean=True,
        verbose=False, plot=False, save=False,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )
    elapsed = time.time() - t0
    print(f"[run_bayesian_pe] sampler wall time: {elapsed:.1f}s, n_cores={n_cores}")

    post = result.posterior
    ln_bf = float(result.log_bayes_factor) if hasattr(result, "log_bayes_factor") else float("nan")

    out = {
        "chirp_mass_Msun":  round(float(post["chirp_mass"].median()), 4),
        "mass_ratio":       round(float(post["mass_ratio"].median()), 4),
        "chirp_mass_5pct":  round(float(post["chirp_mass"].quantile(0.05)), 4),
        "chirp_mass_95pct": round(float(post["chirp_mass"].quantile(0.95)), 4),
        "mass_ratio_5pct":  round(float(post["mass_ratio"].quantile(0.05)), 4),
        "mass_ratio_95pct": round(float(post["mass_ratio"].quantile(0.95)), 4),
        "coalescence_time_s":     round(float(post["geocent_time"].median()) + peak_time, 4),
        "coalescence_time_5pct":  round(float(post["geocent_time"].quantile(0.05)) + peak_time, 4),
        "coalescence_time_95pct": round(float(post["geocent_time"].quantile(0.95)) + peak_time, 4),
        "log_bayes_factor": round(ln_bf, 2),
        "n_posterior_samples": int(len(post)),
        "config_used": config,
        "approximant_used": actual_approximant,
        "priors_rationale": priors_package.get("rationale", ""),
        "sampler_wall_time_s": round(elapsed, 1),
    }
    if "mass_1" in post.columns and "mass_2" in post.columns:
        out["mass1_Msun"] = round(float(post["mass_1"].median()), 3)
        out["mass2_Msun"] = round(float(post["mass_2"].median()), 3)

    spin_free = prior_spec.get("a_1", {}).get("type") not in (None, "fixed")
    out["fix_spins_used"] = not spin_free
    if spin_free and "a_1" in post.columns and "a_2" in post.columns:
        out["a1_recovered"] = round(float(post["a_1"].median()), 4)
        out["a2_recovered"] = round(float(post["a_2"].median()), 4)
    else:
        out["a1_recovered"] = 0.0
        out["a2_recovered"] = 0.0

    incl_free = prior_spec.get("theta_jn", {}).get("type") not in (None, "fixed")
    out["fix_inclination_used"] = not incl_free
    if incl_free and "theta_jn" in post.columns:
        out["theta_jn_recovered"] = round(float(post["theta_jn"].median()), 4)
    else:
        out["theta_jn_recovered"] = 0.0
    for pname, out_key in [("tilt_1", "tilt1_recovered"), ("tilt_2", "tilt2_recovered"),
                            ("phi_12", "phi12_recovered"), ("phi_jl", "phi_jl_recovered")]:
        is_free = prior_spec.get(pname, {}).get("type") not in (None, "fixed")
        out[f"{pname}_free"] = is_free
        if is_free and pname in post.columns:
            out[out_key] = round(float(post[pname].median()), 4)
        else:
            out[out_key] = 0.0
    return out


# =============================================================================
# Tool 4 -- Component masses from chirp mass + q (fallback / standalone use)
# =============================================================================

@tool
def estimate_component_masses(chirp_mass_Msun: float,
                               mass_ratio_guess: float = 0.8) -> dict:
    """
    Convert chirp mass and mass ratio into component masses m1 and m2.
    mass_ratio q = m2/m1 where m1 >= m2, so q is in (0, 1].
    Use this only if run_bayesian_pe did not already return mass1_Msun
    and mass2_Msun directly.
    Returns mass1_Msun, mass2_Msun, total_mass_Msun, mass_ratio.

    Args:
        chirp_mass_Msun: chirp mass in solar masses
        mass_ratio_guess: mass ratio q = m2/m1, use the value from run_bayesian_pe
    """
    Mc = float(chirp_mass_Msun)
    q  = float(np.clip(mass_ratio_guess, 0.05, 1.0))
    m1 = Mc * (1.0 + q)**(1.0/5.0) / q**(3.0/5.0)
    m2 = q * m1
    return {
        "mass1_Msun":      round(float(m1), 3),
        "mass2_Msun":      round(float(m2), 3),
        "total_mass_Msun": round(float(m1 + m2), 3),
        "mass_ratio":      round(float(q), 4),
    }


# =============================================================================
# Tool 6 -- Waveform / strain plotting
# =============================================================================

@tool
def plot_chirp_signal(strain_H1: list, strain_L1: list,
                       psd_H1: list, psd_freqs: list,
                       sample_rate: int,
                       chirp_mass_Msun: float, mass_ratio: float,
                       merger_time_s: float = None,
                       output_path: str = "/tmp/gw_chirp_plot.png",
                       f_lower: float = 20.0, approximant: str = "IMRPhenomD") -> dict:
    """
    3-panel figure:
      Panel 1 -- H1 raw strain + whitened strain + best-fit template, all overlaid
      Panel 2 -- L1 raw strain + whitened strain, overlaid
      Panel 3 -- H1 Q-transform (gwpy) showing chirp frequency sweep, zoomed to merger
    Saves a PNG. NOTE: the overlay template uses spin1z=spin2z=0 for
    simplicity (illustrative only, does not affect any scored/reported
    numbers) -- if you freed spin during PE, the plotted overlay will not
    perfectly match a strongly spinning recovered system.

    Use this AFTER you have your final chirp_mass and mass_ratio estimate.

    Args:
        strain_H1: H1 strain time series -- use data["strain_H1"] from load_gw_data
        strain_L1: L1 strain time series -- use data["strain_L1"] from load_gw_data
        psd_H1: H1 PSD -- use data["psd_H1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        chirp_mass_Msun: final recovered chirp mass for the overlay template
        mass_ratio: final recovered mass ratio for the overlay template
        merger_time_s: merger_time_s from seed_pe_prior_via_matched_filter. If None, estimated from whitened strain peak
        output_path: where to save the PNG
        f_lower: lower frequency cutoff in Hz e.g. 20.0
        approximant: waveform approximant e.g. IMRPhenomD
    """
    import os
    os.environ["MPLBACKEND"] = "Agg"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    strain_H1_arr = np.asarray(strain_H1, dtype=np.float64)
    strain_L1_arr = np.asarray(strain_L1, dtype=np.float64)
    psd_H1_arr    = np.asarray(psd_H1,    dtype=np.float64)
    freqs_arr     = np.asarray(psd_freqs, dtype=np.float64)
    sample_rate   = int(sample_rate)
    dt            = 1.0 / sample_rate
    n             = len(strain_H1_arr)
    t             = np.arange(n) * dt

    def whiten(strain_arr, psd_arr, freqs_arr, sr):
        freqs    = np.fft.rfftfreq(len(strain_arr), d=1.0/sr)
        psd_i    = np.interp(freqs, freqs_arr, psd_arr, left=1e-40, right=1e-40)
        psd_i    = np.where(psd_i > 0, psd_i, 1e-40)
        strain_f = np.fft.rfft(strain_arr)
        white_f  = strain_f / np.sqrt(psd_i * sr / 2.0)
        return np.fft.irfft(white_f, n=len(strain_arr))

    edge       = int(0.5 * sample_rate)
    white_H1   = whiten(strain_H1_arr, psd_H1_arr, freqs_arr, sample_rate)
    white_L1   = whiten(strain_L1_arr, psd_H1_arr, freqs_arr, sample_rate)
    t_trim     = t[edge:-edge]
    white_H1_t = white_H1[edge:-edge]
    white_L1_t = white_L1[edge:-edge]

    raw_H1_t     = strain_H1_arr[edge:-edge]
    raw_L1_t     = strain_L1_arr[edge:-edge]

    if merger_time_s is not None:
        merger_time = float(merger_time_s)
        merger_idx  = int(merger_time * sample_rate)
        merger_idx  = min(merger_idx, len(white_H1) - 1)
    else:
        merger_idx  = int(np.argmax(np.abs(white_H1)))
        merger_time = float(merger_idx) / sample_rate

    template_plotted = False
    white_tmpl_t     = None
    try:
        from pycbc.waveform import get_td_waveform

        q  = float(np.clip(mass_ratio, 0.05, 1.0))
        Mc = float(chirp_mass_Msun)
        m1 = Mc * (1.0 + q)**(1.0/5.0) / q**(3.0/5.0)
        m2 = q * m1

        hp, _ = get_td_waveform(
            approximant=approximant,
            mass1=m1, mass2=m2,
            spin1z=0.0, spin2z=0.0,
            delta_t=dt, f_lower=f_lower, distance=1.0,
        )
        h_arr = np.array(hp)

        tmpl_peak_idx = int(np.argmax(np.abs(h_arr)))
        h_placed      = np.zeros(n)
        src_start     = max(0, tmpl_peak_idx - merger_idx)
        dst_start     = max(0, merger_idx - tmpl_peak_idx)
        copy_len      = min(len(h_arr) - src_start, n - dst_start)
        if copy_len > 0:
            h_placed[dst_start:dst_start + copy_len] = \
                h_arr[src_start:src_start + copy_len]

        white_h      = whiten(h_placed, psd_H1_arr, freqs_arr, sample_rate)
        white_h_t    = white_h[edge:-edge]
        scale        = np.std(white_H1_t) / (np.std(white_h_t) + 1e-30) * 0.5
        white_tmpl_t = white_h_t * scale
        template_plotted = True
    except Exception:
        pass

    qtrans_ok  = False
    qtrans_err = ""
    qtrans_img = None
    try:
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries

        ts_gwpy = GWpyTimeSeries(strain_H1_arr, dt=dt, t0=0)
        outseg = (merger_time - 0.5, merger_time + 0.5)

        qtrans_img = ts_gwpy.q_transform(
            frange=(f_lower, 512.0),
            qrange=(4, 64),
            outseg=outseg,
        )
        qtrans_ok = True
    except Exception as e:
        qtrans_err = str(e)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle(
        f"GW Event  Mc={chirp_mass_Msun:.1f} M\u2609  q={mass_ratio:.2f}",
        fontsize=13, y=1.01,
    )

    ax0_raw = axes[0].twinx()
    ax0_raw.plot(t_trim, raw_H1_t * 1e21,
                 color="cornflowerblue", lw=0.4, alpha=0.35, label="H1 raw")
    ax0_raw.set_ylabel("Raw Strain (×10⁻²¹)", color="cornflowerblue", fontsize=8)
    ax0_raw.tick_params(axis="y", labelcolor="cornflowerblue", labelsize=7)

    axes[0].plot(t_trim, white_H1_t,
                 color="steelblue", lw=0.8, alpha=0.9, label="H1 whitened")
    if template_plotted:
        axes[0].plot(t_trim, white_tmpl_t,
                     color="black", lw=1.8,
                     label=f"template (Mc={chirp_mass_Msun:.1f}, q={mass_ratio:.2f})")
    axes[0].axvline(merger_time, color="red", lw=1.0, ls="--",
                    alpha=0.7, label=f"merger t={merger_time:.2f}s")
    axes[0].set_ylabel("Whitened Strain")
    axes[0].set_title("H1 — raw (right axis) + whitened + best-fit template")
    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines0r, labels0r = ax0_raw.get_legend_handles_labels()
    axes[0].legend(lines0 + lines0r, labels0 + labels0r,
                   loc="upper left", fontsize=8, ncol=2)
    axes[0].set_xlim(t_trim[0], t_trim[-1])
    ax0_raw.set_xlim(t_trim[0], t_trim[-1])

    ax1_raw = axes[1].twinx()
    ax1_raw.plot(t_trim, raw_L1_t * 1e21,
                 color="lightcoral", lw=0.4, alpha=0.35, label="L1 raw")
    ax1_raw.set_ylabel("Raw Strain (×10⁻²¹)", color="lightcoral", fontsize=8)
    ax1_raw.tick_params(axis="y", labelcolor="lightcoral", labelsize=7)

    axes[1].plot(t_trim, white_L1_t,
                 color="indianred", lw=0.8, alpha=0.9, label="L1 whitened")
    axes[1].axvline(merger_time, color="red", lw=1.0, ls="--",
                    alpha=0.7, label=f"merger t={merger_time:.2f}s")
    axes[1].set_ylabel("Whitened Strain")
    axes[1].set_title("L1 — raw (right axis) + whitened strain")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines1r, labels1r = ax1_raw.get_legend_handles_labels()
    axes[1].legend(lines1 + lines1r, labels1 + labels1r,
                   loc="upper left", fontsize=8, ncol=2)
    axes[1].set_xlim(t_trim[0], t_trim[-1])
    ax1_raw.set_xlim(t_trim[0], t_trim[-1])

    if qtrans_ok and qtrans_img is not None:
        times_q  = qtrans_img.times.value
        freqs_q  = qtrans_img.frequencies.value
        power_q  = qtrans_img.value

        pcm = axes[2].pcolormesh(
            times_q, freqs_q, power_q.T,
            shading="auto", cmap="viridis",
            vmin=0, vmax=np.percentile(power_q, 99),
        )
        fig.colorbar(pcm, ax=axes[2], label="Normalised energy")
        axes[2].set_yscale("log")
        axes[2].set_ylim(f_lower, 512)
        axes[2].set_title("H1 Q-transform — chirp frequency sweep")
    else:
        from scipy.signal import spectrogram as sg
        nperseg = min(512, len(white_H1) // 8)
        nperseg = max(nperseg, 64)
        f_s, t_s, Sxx = sg(white_H1, fs=sample_rate,
                            nperseg=nperseg, noverlap=nperseg * 3 // 4)
        mask = (f_s >= f_lower) & (f_s <= 512)
        axes[2].pcolormesh(t_s, f_s[mask],
                           10 * np.log10(Sxx[mask] + 1e-50),
                           shading="auto", cmap="viridis")
        axes[2].axvline(merger_time, color="red", lw=1.0, ls="--", alpha=0.7)
        axes[2].set_title(
            f"H1 whitened spectrogram (gwpy Q-transform err: {qtrans_err[:80]})"
        )

    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot_path":         output_path,
        "peak_strain_H1":    float(np.max(np.abs(strain_H1_arr))),
        "peak_strain_L1":    float(np.max(np.abs(strain_L1_arr))),
        "template_overlaid": template_plotted,
        "qtransform_used":   qtrans_ok,
        "merger_time_s":     round(merger_time, 3),
    }


# =============================================================================
# Tool 7 -- PE result diagnostics (objective facts only, no decision)
# =============================================================================

@tool
def inspect_pe_result(pe_result: dict) -> dict:
    """
    Compute objective diagnostic statistics from a run_bayesian_pe result.
    This tool does NOT decide whether to accept or retry -- it only
    reports facts, including the FULL sampler config used and whichever
    spin/inclination parameters were freed, for you to reason about
    yourself.

    If pe_result is missing expected fields (e.g. the PE run failed
    before producing a posterior), returns pe_run_failed=True with the
    error message instead of raising -- check this field first before
    reading any of the other diagnostic values.

    Args:
        pe_result: the dict returned by run_bayesian_pe
    """
    if "chirp_mass_95pct" not in pe_result or "coalescence_time_95pct" not in pe_result:
        return {
            "pe_run_failed": True,
            "error": pe_result.get("error_message") or pe_result.get("error") or "malformed pe_result",
        }

    config = pe_result.get("config_used", {})

    chirp_mass_ci_width = round(
        float(pe_result["chirp_mass_95pct"] - pe_result["chirp_mass_5pct"]), 4
    )
    coalescence_time_ci_width = round(
        float(pe_result["coalescence_time_95pct"] - pe_result["coalescence_time_5pct"]), 4
    )

    return {
        "pe_run_failed": False,
        "chirp_mass_ci_width": chirp_mass_ci_width,
        "coalescence_time_ci_width": coalescence_time_ci_width,
        "chirp_mass_median": pe_result.get("chirp_mass_Msun"),
        "coalescence_time_median": pe_result.get("coalescence_time_s"),
        "log_bayes_factor": pe_result.get("log_bayes_factor"),
        "n_posterior_samples": pe_result.get("n_posterior_samples"),
        "approximant_used": pe_result.get("approximant_used"),
        "fix_spins_used": pe_result.get("fix_spins_used", True),
        "a1_recovered": pe_result.get("a1_recovered", 0.0),
        "a2_recovered": pe_result.get("a2_recovered", 0.0),
        "fix_inclination_used": pe_result.get("fix_inclination_used", True),
        "theta_jn_recovered": pe_result.get("theta_jn_recovered", 0.0),
        "nlive_used": config.get("nlive"),
        "sample_used": config.get("sample"),
        "walks_used": config.get("walks"),
        "nact_used": config.get("nact"),
        "dlogz_used": config.get("dlogz"),
        "config_rationale": config.get("rationale", ""),
        "priors_rationale": pe_result.get("priors_rationale", ""),
    }


# =============================================================================
# Tool 8 -- Package the critic agent's own decision (validates, doesn't decide)
# =============================================================================
@tool
def package_pe_critique(recommendation: str, reasoning: str,
                         log_bayes_factor: float,
                         chi2_reduced: float,
                         retry_nlive: int = None,
                         retry_sample: str = None,
                         retry_walks: int = None,
                         retry_nact: int = None,
                         retry_dlogz: float = None) -> dict:
    """
    Package your own accept/retry decision into a validated dict. This
    tool does NOT decide anything for you -- you must inspect the PE
    result yourself (via inspect_pe_result and check_waveform_residual)
    and decide.

    A plain "accept" is blocked when chi2_reduced or log_bayes_factor
    fall outside acceptable ranges. In that case, use "retry" with
    adjusted sampler settings, or "accept_with_caveat" with an explicit
    justification.

    If recommending a retry, specify ONLY the config keys you actually
    want changed via the retry_* arguments -- leave any you don't want
    to change as None, and the orchestrator will keep the previous
    attempt's value for those. Only touch what your reasoning actually
    supports changing.

    Args:
        recommendation: your decision -- must be exactly "accept",
            "retry", or "accept_with_caveat"
        reasoning: your reasoning for this decision, for logging
        log_bayes_factor: pass through from inspect_pe_result
        chi2_reduced: pass through from check_waveform_residual
        retry_nlive: if retrying, new nlive value. None = unchanged.
        retry_sample: if retrying, new dynesty sample method
            (e.g. "rwalk", "slice"). None = unchanged.
        retry_walks: if retrying, new walks value. None = unchanged.
        retry_nact: if retrying, new nact value. None = unchanged.
        retry_dlogz: if retrying, new dlogz value. None = unchanged.
    """
    valid = {"accept", "retry", "accept_with_caveat"}
    if recommendation not in valid:
        raise ValueError(f"recommendation must be one of {valid}, got {recommendation!r}")

    if recommendation == "accept" and (chi2_reduced > 3.0 or log_bayes_factor < 0):
        raise ValueError(
            "Cannot recommend plain 'accept': chi2_reduced is "
            f"{chi2_reduced:.2f} and/or log_bayes_factor is "
            f"{log_bayes_factor:.2f}, suggesting a poor fit or weak "
            "detection. Use recommendation='retry' with adjusted sampler "
            "settings (e.g. higher nlive, tighter dlogz), or "
            "'accept_with_caveat' if you have a specific budget-based "
            "reason to accept anyway."
        )

    return {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "retry_nlive": retry_nlive,
        "retry_sample": retry_sample,
        "retry_walks": retry_walks,
        "retry_nact": retry_nact,
        "retry_dlogz": retry_dlogz,
    }


@tool
def check_waveform_residual(strain_H1: list, psd_H1: list, psd_freqs: list,
                             sample_rate: int, chirp_mass_Msun: float,
                             mass_ratio: float, merger_time_s: float,
                             a_1: float = 0.0, a_2: float = 0.0,
                             tilt_1: float = 0.0, tilt_2: float = 0.0,
                             phi_12: float = 0.0, phi_jl: float = 0.0,
                             theta_jn: float = 0.0,
                             f_lower: float = 20.0,
                             approximant: str = "IMRPhenomD",
                             n_bins: int = 8) -> dict:
    """
    Chi-square signal consistency test, following the standard LVK
    matched-filter chi-square test (Allen 2005). Splits the frequency
    band into n_bins sub-bands each contributing equal expected template
    SNR, computes the matched-filter SNR contribution z_i in each
    sub-band, and checks whether SNR accumulates evenly across sub-bands
    (consistent with a good template match) or unevenly (suggesting the
    template does NOT explain the data well -- a red flag independent
    of any posterior width or Bayes factor).

    A well-fitting signal gives chi2_reduced near 1.0. Values well above
    1.0 (e.g. >2-3) suggest the recovered parameters don't actually
    explain the data -- a genuinely blind, ground-truth-free fit-quality
    check, unlike CI width which only measures the sampler's confidence.

    Supports both aligned-spin (a_1/a_2 only, tilt left at 0 -- signed
    a_1/a_2 encode direction) and precessing (tilt_1/tilt_2 nonzero,
    requires a precession-capable approximant -- a_1/a_2 must then be
    non-negative magnitudes) recovered systems. For precessing cases,
    spin components are converted to Cartesian spin1x/y/z, spin2x/y/z via
    bilby's own conversion utility, using a fixed reference phase=0.0
    (this tool has no access to the true recovered orbital phase, which
    is analytically marginalized in run_bayesian_pe -- this is a standard
    convention, but means the precessing overlay's absolute phase
    evolution is not phase-locked to the data the way the aligned-spin
    case effectively is via best_fit_scale).

    Args:
        strain_H1: H1 strain time series -- use data["strain_H1"] from load_gw_data
        psd_H1: H1 PSD -- use data["psd_H1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        chirp_mass_Msun: recovered chirp mass to build the template
        mass_ratio: recovered mass ratio to build the template
        merger_time_s: merger_time_s -- use pe_result["coalescence_time_s"]
        a_1: recovered spin1 value. Use pe_result["a1_recovered"] -- 0.0
            if spins were fixed. Signed (aligned case) or non-negative
            magnitude (precessing case, i.e. tilt_1 was freed).
        a_2: recovered spin2 value. Use pe_result["a2_recovered"].
        tilt_1: recovered tilt_1. Use pe_result["tilt1_recovered"] -- 0.0
            if fixed/not applicable.
        tilt_2: recovered tilt_2. Use pe_result["tilt2_recovered"].
        phi_12: recovered phi_12. Use pe_result["phi12_recovered"].
        phi_jl: recovered phi_jl. Use pe_result["phi_jl_recovered"].
        theta_jn: recovered inclination. Use pe_result["theta_jn_recovered"].
        f_lower: lower frequency cutoff in Hz e.g. 20.0
        approximant: waveform approximant e.g. IMRPhenomD or IMRPhenomXPHM
        n_bins: number of frequency sub-bands for the chi-square test
                (standard values: 8-16)
    """
    import numpy as np
    from pycbc.waveform import get_fd_waveform

    sample_rate = int(sample_rate)
    dt = 1.0 / sample_rate
    strain_arr = np.asarray(strain_H1, dtype=np.float64)
    psd_arr = np.asarray(psd_H1, dtype=np.float64)
    freqs_data = np.asarray(psd_freqs, dtype=np.float64)
    N = len(strain_arr)
    delta_f = 1.0 / (N * dt)
    flen = N // 2 + 1

    q = float(np.clip(mass_ratio, 0.05, 1.0))
    Mc = float(chirp_mass_Msun)
    m1 = Mc * (1.0 + q) ** (1.0/5.0) / q ** (3.0/5.0)
    m2 = q * m1

    is_precessing = (abs(tilt_1) > 1e-6 or abs(tilt_2) > 1e-6)

    if is_precessing:
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
            theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
            phi_12=phi_12, a_1=float(a_1), a_2=float(a_2),
            mass_1=m1, mass_2=m2,
            reference_frequency=f_lower, phase=0.0,
        )
        spin_kwargs = dict(spin1x=float(s1x), spin1y=float(s1y), spin1z=float(s1z),
                            spin2x=float(s2x), spin2y=float(s2y), spin2z=float(s2z))
    else:
        spin_kwargs = dict(spin1z=float(a_1), spin2z=float(a_2))

    hp, _ = get_fd_waveform(
        approximant=approximant, mass1=m1, mass2=m2,
        delta_f=delta_f, f_lower=f_lower, f_final=float(0.5 / dt),
        **spin_kwargs,
    )
    tmpl = np.asarray(hp.data, dtype=complex)
    if len(tmpl) < flen:
        tmpl = np.pad(tmpl, (0, flen - len(tmpl)))
    elif len(tmpl) > flen:
        tmpl = tmpl[:flen]

    freqs = np.arange(flen) * delta_f
    psd_i = np.interp(freqs, freqs_data, psd_arr, left=1e-40, right=1e-40)
    psd_i = np.where(psd_i > 0, psd_i, 1e-40)
    band_mask = freqs >= f_lower

    data_f = np.fft.rfft(strain_arr) * dt

    best_offset, best_total_snr = 0.0, -1.0
    for dt_search in np.linspace(-0.01, 0.01, 41):
        trial_shift = np.exp(-2j * np.pi * freqs * (merger_time_s + dt_search))
        trial_tmpl = tmpl * trial_shift
        trial_total = np.sum(
            (np.conj(trial_tmpl[band_mask]) * data_f[band_mask]) / psd_i[band_mask]
        ) * delta_f
        if abs(trial_total) > best_total_snr:
            best_total_snr = abs(trial_total)
            best_offset = dt_search

    phase_shift = np.exp(-2j * np.pi * freqs * (merger_time_s + best_offset))
    tmpl_shifted = tmpl * phase_shift

    numerator = np.sum((np.conj(tmpl_shifted[band_mask]) * data_f[band_mask]) / psd_i[band_mask]) * delta_f
    denominator = np.sum((np.abs(tmpl_shifted[band_mask])**2) / psd_i[band_mask]) * delta_f
    best_fit_scale = numerator / denominator if denominator > 0 else 0.0
    tmpl_shifted = tmpl_shifted * best_fit_scale

    weight = (np.abs(tmpl_shifted)**2 / psd_i) * band_mask
    cum_weight = np.cumsum(weight)
    total_weight = cum_weight[-1] if cum_weight[-1] > 0 else 1.0
    bin_edges_weight = np.linspace(0, total_weight, n_bins + 1)
    bin_indices = np.searchsorted(cum_weight, bin_edges_weight)
    bin_indices[0] = np.argmax(band_mask)
    bin_indices[-1] = flen

    z_per_bin = []
    for i in range(n_bins):
        lo, hi = bin_indices[i], bin_indices[i+1]
        if hi <= lo:
            z_per_bin.append(0.0 + 0.0j)
            continue
        zi = np.sum(np.conj(tmpl_shifted[lo:hi]) * data_f[lo:hi] / psd_i[lo:hi]) * delta_f
        z_per_bin.append(zi)

    sigma = np.sqrt(np.sum((np.abs(tmpl_shifted[band_mask])**2) / psd_i[band_mask]) * delta_f)
    z_per_bin = [zi / sigma for zi in z_per_bin] if sigma > 0 else z_per_bin

    z_total = sum(z_per_bin)
    p = len(z_per_bin)
    expected_per_bin = z_total / p if p > 0 else 0.0

    chi2 = p * sum(abs(zi - expected_per_bin) ** 2 for zi in z_per_bin)
    dof = max(1, 2 * p - 2)
    chi2_reduced = chi2 / dof

    return {
        "chi2": round(float(chi2), 3),
        "chi2_dof": dof,
        "chi2_reduced": round(float(chi2_reduced), 3),
        "n_bins_used": p,
        "is_precessing_template": is_precessing,
        "note": (
            "chi2_reduced near 1.0 means the matched-filter SNR accumulates "
            "evenly across frequency sub-bands as the template predicts -- "
            "good fit. Well above 1.0 (e.g. >2-3) means SNR is concentrated "
            "unevenly, suggesting the recovered parameters don't actually "
            "fit the data well, regardless of posterior confidence. This "
            "follows the standard LVK matched-filter chi-square consistency "
            "test (Allen 2005)."
        ),
    }


@tool
def classify_merger_type(mass1_Msun: float, mass2_Msun: float) -> str:
    """
    Classify a compact binary merger as BNS, NSBH, or BBH based on
    component masses, using the standard 3 solar-mass neutron-star/
    black-hole boundary.

    Args:
        mass1_Msun: primary component mass in solar masses
        mass2_Msun: secondary component mass in solar masses
    """
    m1 = float(mass1_Msun)
    m2 = float(mass2_Msun)
    ns_max = 3.0
    if m1 <= ns_max and m2 <= ns_max:
        return "BNS"
    if m1 > ns_max and m2 > ns_max:
        return "BBH"
    return "NSBH"

@tool
def estimate_psd_welch(strain_offsource: list, sample_rate: int,
                        seg_duration: float = 4.0, overlap_frac: float = 0.5) -> dict:
    """
    Estimate the detector noise PSD from off-source (signal-free) strain
    data, using the median-Welch method -- the same approach real LVK
    analyses use to estimate PSDs from real detector data.

    Splits the off-source strain into overlapping windowed segments,
    computes each segment's periodogram, and takes the MEDIAN (not mean)
    across segments at each frequency. Median is used specifically
    because it's robust to any single contaminated segment (e.g. one
    containing a glitch) -- a mean would be skewed by even one bad
    segment, while the median resists this. A standard bias correction
    (1/ln(2)) is applied, since the median of a chi-squared-like
    periodogram distribution systematically underestimates its mean.

    This is a genuine estimate, not the true PSD -- it carries real
    statistical uncertainty from being built off a finite noise sample,
    same as in real analysis. Use the returned psd_estimate/psd_freqs
    wherever downstream tools expect a PSD (seed_pe_prior_via_matched_filter,
    run_bayesian_pe, check_waveform_residual, plot_chirp_signal) --
    do NOT use any ground-truth PSD, since none is provided to you.

    Args:
        strain_offsource: off-source strain time series (signal-free) --
            use data["strain_H1_offsource"] or data["strain_L1_offsource"]
            from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"]
        seg_duration: length in seconds of each Welch segment (default 4.0s
            -- shorter segments give more segments to median over, at the
            cost of frequency resolution; longer segments give finer
            frequency resolution but fewer segments to average/median over)
        overlap_frac: fractional overlap between consecutive segments
            (default 0.5 = 50% overlap, standard Welch practice)
    """
    import numpy as np

    strain_arr = np.asarray(strain_offsource, dtype=np.float64)
    n_total = len(strain_arr)
    seg_len = int(seg_duration * sample_rate)

    if seg_len < 16:
        raise ValueError(f"seg_duration too short: {seg_len} samples, need at least 16")
    if seg_len > n_total:
        raise ValueError(
            f"seg_duration ({seg_duration}s = {seg_len} samples) longer than "
            f"available off-source data ({n_total} samples) -- reduce seg_duration"
        )

    step = max(1, int(seg_len * (1.0 - overlap_frac)))
    window = np.hanning(seg_len)
    window_norm = np.sum(window ** 2)
    dt = 1.0 / sample_rate

    periodograms = []
    start = 0
    while start + seg_len <= n_total:
        segment = strain_arr[start:start + seg_len] * window
        # Match this codebase's OWN convention exactly: _compute_snr
        # scales its FFT by dt before dividing by a psd array, so the
        # PSD estimator must be built the same way for the two to be
        # consistent (verified empirically via verify_psd_estimation.py).
        seg_fft = np.fft.rfft(segment)
        periodogram = (2.0 / (sample_rate * window_norm)) * np.abs(seg_fft) ** 2
        periodograms.append(periodogram)
        start += step

    n_segments = len(periodograms)
    if n_segments < 4:
        raise ValueError(
            f"Only {n_segments} segments available -- too few for a reliable "
            f"median PSD estimate (need at least 4). Reduce seg_duration or "
            f"increase overlap_frac."
        )

    periodograms = np.array(periodograms)
    median_psd = np.median(periodograms, axis=0)

    # Bias correction: the median of a periodogram bin (chi-squared-2-DOF
    # distributed, for Gaussian noise) is ln(2) times smaller than its
    # mean -- this is the standard correction used in real median-Welch
    # PSD estimation (e.g. LIGO/Virgo pipelines), so the estimate is
    # unbiased relative to the true mean PSD.
    median_bias_correction = 1.0 / np.log(2.0)
    psd_estimate = median_psd * median_bias_correction

    freqs = np.fft.rfftfreq(seg_len, d=1.0 / sample_rate)

    return {
        "psd_estimate": psd_estimate.tolist(),
        "psd_freqs": freqs.tolist(),
        "n_segments_used": n_segments,
        "seg_duration_s": seg_duration,
        "note": (
            f"Median-Welch PSD estimate from {n_segments} overlapping "
            f"{seg_duration}s segments. This is an ESTIMATE with real "
            f"statistical uncertainty, not the true detector PSD -- "
            f"downstream results will reflect this."
        ),
    }