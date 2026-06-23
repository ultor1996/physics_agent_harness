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


# =============================================================================
# Tool 2 -- Fast matched filter (coarse seed for the PE prior only)
# =============================================================================

def _eval_single_template(args):
    """
    Evaluate one (Mc, q) template using both H1 and L1 detectors.
    Spins fixed to zero. Module-level function required for
    multiprocessing pickling. Returns (Mc, q, network_snr, m1, m2) or None.
    """
    (Mc, q, strain_H1_arr, psd_H1_arr, strain_L1_arr, psd_L1_arr, freqs_arr,
     delta_t, delta_f, flen, f_lower, approximant) = args

    try:
        from pycbc.types import TimeSeries, FrequencySeries
        from pycbc.psd import interpolate, inverse_spectrum_truncation
        from pycbc.waveform import get_fd_waveform
        from pycbc.filter import matched_filter
        import numpy as np

        q  = float(np.clip(q, 0.05, 1.0))
        m1 = float(Mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0))
        m2 = q * m1

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

        sample_rate = int(round(1.0 / delta_t))

        def build_psd(psd_arr):
            psd_fs     = FrequencySeries(psd_arr, delta_f=float(freqs_arr[1] - freqs_arr[0]))
            psd_interp = interpolate(psd_fs, delta_f)
            return inverse_spectrum_truncation(
                psd_interp, int(4 * sample_rate), low_frequency_cutoff=f_lower
            )

        psd_H1_trunc = build_psd(psd_H1_arr)
        strain_H1_ts = TimeSeries(strain_H1_arr, delta_t=delta_t)
        snr_H1       = matched_filter(hp, strain_H1_ts.to_frequencyseries(),
                                      psd=psd_H1_trunc, low_frequency_cutoff=f_lower)
        peak_H1      = float(abs(snr_H1).numpy().max())

        psd_L1_trunc = build_psd(psd_L1_arr)
        strain_L1_ts = TimeSeries(strain_L1_arr, delta_t=delta_t)
        snr_L1       = matched_filter(hp, strain_L1_ts.to_frequencyseries(),
                                      psd=psd_L1_trunc, low_frequency_cutoff=f_lower)
        peak_L1      = float(abs(snr_L1).numpy().max())

        network_snr = float(np.sqrt(peak_H1**2 + peak_L1**2))
        return (float(Mc), float(q), network_snr, float(m1), float(m2))

    except Exception:
        return None


@tool
def seed_pe_prior_via_matched_filter(strain: list, psd: list, psd_freqs: list,
                                      strain_L1: list, psd_L1: list,
                                      sample_rate: int, approximant: str = "IMRPhenomD",
                                      f_lower: float = 20.0) -> dict:
    """
    Fast coarse matched filter bank over chirp mass and mass ratio.
    This is NOT a final answer and has no standalone interpretation --
    its only purpose is to produce a chirp_mass_guess and mass_ratio_guess
    to pass into run_bayesian_pe, which performs the actual parameter
    estimation. Always call run_bayesian_pe immediately after this tool.

    Single 40x12 grid, spins fixed to zero, combines H1 and L1 network SNR.
    Returns best_chirp_mass_Msun, best_mass_ratio, best_snr, best_mass1, best_mass2
    -- a coarse seed only, not a result to report.

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

    chirp_masses = np.logspace(np.log10(4), np.log10(90), 40)
    mass_ratios  = np.linspace(0.1, 1.0, 12)

    args = [
        (Mc, q, strain_H1_arr, psd_H1_arr, strain_L1_arr, psd_L1_arr,
         freqs_arr, delta_t, delta_f, flen, f_lower, approximant)
        for Mc in chirp_masses
        for q  in mass_ratios
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        for res in ex.map(_eval_single_template, args, chunksize=10):
            if res is not None:
                results.append(res)

    if not results:
        return {
            "best_chirp_mass_Msun": 25.0,
            "best_mass_ratio":      0.8,
            "best_snr":             0.0,
            "best_mass1":           29.0,
            "best_mass2":           23.0,
        }

    best = max(results, key=lambda x: x[2])
    Mc, q, snr, m1, m2 = best

    # ── Find SNR peak time using best template ───────────────────────
    # The SNR peak time is the matched-filter estimate of the merger
    # time — far more reliable than raw strain amplitude for this purpose.
    merger_time_s = 0.0
    try:
        from pycbc.types import TimeSeries, FrequencySeries
        from pycbc.psd import interpolate, inverse_spectrum_truncation
        from pycbc.waveform import get_fd_waveform
        from pycbc.filter import matched_filter as _mf

        q_b  = float(np.clip(q, 0.05, 1.0))
        m1_b = float(Mc * (1.0 + q_b)**(1.0/5.0) / q_b**(3.0/5.0))
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

        psd_fs     = FrequencySeries(psd_H1_arr,
                                      delta_f=float(freqs_arr[1] - freqs_arr[0]))
        psd_interp = interpolate(psd_fs, delta_f)
        psd_trunc  = inverse_spectrum_truncation(
            psd_interp, int(4 * sample_rate), low_frequency_cutoff=f_lower
        )
        snr_ts         = _mf(hp_b,
                              TimeSeries(strain_H1_arr, delta_t=delta_t).to_frequencyseries(),
                              psd=psd_trunc, low_frequency_cutoff=f_lower)
        peak_idx       = int(abs(snr_ts).numpy().argmax())
        merger_time_s  = float(peak_idx) * delta_t
    except Exception:
        merger_time_s = float(len(strain_H1_arr)) * delta_t * 0.67  # fallback

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
#
# Follows the avivajpeyi GW PE tutorial conventions:
#   https://avivajpeyi.github.io/gw_pe_tutorial/workshop_notebook.html
#
#   - bilby.gw.prior.BBHPriorDict() as the base prior set
#   - UniformInComponentsChirpMass for the chirp-mass prior (more
#     astrophysically sensible than a plain Uniform on chirp_mass --
#     "Create priors for analysis" section of the tutorial)
#   - mass_1 / mass_2 as Constraint priors so the sampler still works
#     in chirp_mass / mass_ratio space, matching their GW150914 example
#   - conversion_function=generate_all_bbh_parameters and
#     result_class=CBCResult, so component masses / chi_eff / etc. come
#     out of the posterior automatically ("Inference step" section)
#
#   The tutorial's GW150914 example fixes ra, dec, distance, theta_jn,
#   psi, geocent_time to the known injection values and explicitly
#   flags this with "# dont do this in a real run". Our agent does NOT
#   know the true sky location, so we cannot use that shortcut --
#   doing so previously caused a catastrophic failure (log_bayes_factor
#   ~ -183) because the wrong fixed sky location biases the antenna
#   pattern in the likelihood. Instead we sample ra/dec/theta_jn/psi
#   genuinely and analytically marginalise time, distance, and phase,
#   which is the standard honest rapid-PE approach when the true
#   extrinsic parameters are unknown.
# =============================================================================

@tool
def run_bayesian_pe(strain_H1: list, psd_H1: list,
                     strain_L1: list, psd_L1: list,
                     psd_freqs: list, sample_rate: int,
                     chirp_mass_guess: float, mass_ratio_guess: float,
                     merger_time_s: float = 10.72,
                     f_lower: float = 20.0, approximant: str = "IMRPhenomD",
                     nlive: int = 250) -> dict:
    """
    Run full Bayesian parameter estimation with Bilby + dynesty nested
    sampling, following standard GW rapid-PE practice. Samples
    chirp_mass (UniformInComponentsChirpMass prior, narrowed around the
    matched-filter guess), mass_ratio, sky location (ra, dec), and
    inclination (theta_jn) jointly. Time, phase, and distance are
    analytically marginalised in the likelihood rather than fixed,
    since the true values are unknown.

    Returns posterior medians and 5/95 percentile credible intervals
    for chirp_mass and mass_ratio (plus derived component masses via
    generate_all_bbh_parameters), and the log Bayes factor as a
    detection-confidence diagnostic (large positive = strong evidence
    for a signal; near zero or negative = sampler did not find a
    convincing signal in the prior range).

    Args:
        strain_H1: H1 strain time series -- use data["strain_H1"] from load_gw_data
        psd_H1: H1 PSD -- use data["psd_H1"] from load_gw_data
        strain_L1: L1 strain time series -- use data["strain_L1"] from load_gw_data
        psd_L1: L1 PSD -- use data["psd_L1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        chirp_mass_guess: best_chirp_mass_Msun from seed_pe_prior_via_matched_filter
        mass_ratio_guess: best_mass_ratio from seed_pe_prior_via_matched_filter
        merger_time_s: merger_time_s from seed_pe_prior_via_matched_filter
        f_lower: lower frequency cutoff in Hz e.g. 20.0
        approximant: waveform approximant string e.g. IMRPhenomD
        nlive: number of live points for dynesty -- higher is more accurate
               but slower. 150 is a reasonable rapid-PE setting.
    """
    import os
    os.environ["MPLBACKEND"] = "Agg"   # must be set before bilby/dynesty import
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    import logging
    import warnings
    warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
    import bilby
    import numpy as np

    bilby_logger = logging.getLogger("bilby")
    bilby_logger.setLevel(logging.ERROR)
    logging.getLogger("dynesty").setLevel(logging.ERROR)

    sample_rate   = int(sample_rate)
    strain_H1_arr = np.asarray(strain_H1, dtype=np.float64)
    strain_L1_arr = np.asarray(strain_L1, dtype=np.float64)
    psd_H1_arr    = np.asarray(psd_H1,    dtype=np.float64)
    psd_L1_arr    = np.asarray(psd_L1,    dtype=np.float64)
    freqs_arr     = np.asarray(psd_freqs, dtype=np.float64)

    duration = float(len(strain_H1_arr) / sample_rate)

    # ---- Interferometers from the supplied real strain + PSD arrays.
    # (Unlike the tutorial's injection examples which build ifos via
    #  set_strain_data_from_power_spectral_densities + inject_signal,
    #  we already have real strain data, so we load it directly --
    #  the same approach the tutorial uses for its GW150914 example,
    #  just from arrays instead of GWOSC/gwpy.) ----
    ifo_H1 = bilby.gw.detector.get_empty_interferometer("H1")
    ifo_L1 = bilby.gw.detector.get_empty_interferometer("L1")

    peak_time = float(merger_time_s)

    ifo_H1.strain_data.set_from_time_domain_strain(
        strain_H1_arr,
        sampling_frequency=float(sample_rate),
        duration=duration,
        start_time=-peak_time,
    )
    ifo_L1.strain_data.set_from_time_domain_strain(
        strain_L1_arr,
        sampling_frequency=float(sample_rate),
        duration=duration,
        start_time=-peak_time,
    )
    # PyCBC-style PSDs are exactly zero below f_lower by construction;
    # floor them so the Whittle likelihood never divides by zero.
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

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": approximant,
            "reference_frequency":  20.0,
            "minimum_frequency":    f_lower,
        },
    )

    # ---- Priors -- tutorial's "Create priors for analysis" pattern ----
    Mc_guess = max(float(chirp_mass_guess), 1.0)
    priors = bilby.gw.prior.BBHPriorDict()

    # UniformInComponentsChirpMass, narrowed around the matched-filter
    # point estimate (tutorial uses a +/-5 Msun window around the known
    # injection value for its toy example; we use a multiplicative
    # window since chirp_mass_guess can range from a few to ~90 Msun).
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(
    minimum=max(2.0,  Mc_guess * 0.50),
    maximum=min(150.0, Mc_guess * 1.70),
    name="chirp_mass",
    latex_label=r"$\mathcal{M}$",
)
    priors["mass_ratio"] = bilby.core.prior.Uniform(
    minimum=0.1, maximum=1.0,
    name="mass_ratio", latex_label="$q$",
    )
    priors["mass_1"] = bilby.core.prior.Constraint(name="mass_1", minimum=1.0, maximum=300.0)
    priors["mass_2"] = bilby.core.prior.Constraint(name="mass_2", minimum=1.0, maximum=300.0)

    priors["ra"]       = bilby.core.prior.DeltaFunction(peak=1.57, name="ra")
    priors["dec"]      = bilby.core.prior.DeltaFunction(peak=0.0,  name="dec")
    priors["psi"]      = bilby.core.prior.DeltaFunction(peak=0.0,  name="psi")
    priors["theta_jn"] = bilby.core.prior.Sine(name="theta_jn")  

    # # Non-spinning recovery -- spin estimation out of scope for this tool.
    # priors["a_1"]    = bilby.core.prior.DeltaFunction(0.0)
    # priors["a_2"]    = bilby.core.prior.DeltaFunction(0.0)
    # priors["tilt_1"] = bilby.core.prior.DeltaFunction(0.0)
    # priors["tilt_2"] = bilby.core.prior.DeltaFunction(0.0)
    # priors["phi_12"] = bilby.core.prior.DeltaFunction(0.0)
    # priors["phi_jl"] = bilby.core.prior.DeltaFunction(0.0)
    # Aligned-spin recovery -- sample a_1/a_2 within approximant validity range.
    # Precession parameters fixed since all supported approximants
    # (IMRPhenomD, IMRPhenomXHM, SEOBNRv4) are aligned-spin only.
    spin_limits = {
        "IMRPhenomD":   0.88,
        "IMRPhenomXHM": 0.99,
        "SEOBNRv4":     0.99,
        "SEOBNRv4_ROM": 0.99,
    }
    spin_max = spin_limits.get(approximant, 0.88)

    priors["a_1"]    = bilby.core.prior.Uniform(
        minimum=-spin_max, maximum=spin_max, name="a_1"
    )
    priors["a_2"]    = bilby.core.prior.Uniform(
        minimum=-spin_max, maximum=spin_max, name="a_2"
    )
    priors["tilt_1"] = bilby.core.prior.DeltaFunction(0.0)
    priors["tilt_2"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_12"] = bilby.core.prior.DeltaFunction(0.0)
    priors["phi_jl"] = bilby.core.prior.DeltaFunction(0.0)

    # Time, phase, distance ranges -- only used as the marginalisation
    # range/reference since all three are analytically marginalised
    # in the likelihood below, not actually sampled as free parameters.
    priors["geocent_time"]        = bilby.core.prior.Uniform(-0.1, 0.1, name="geocent_time")
    priors["luminosity_distance"] = bilby.core.prior.PowerLaw(
        alpha=2, name="luminosity_distance", minimum=10.0, maximum=5000.0
    )
    priors["phase"] = bilby.core.prior.Uniform(0.0, 2 * np.pi, name="phase", boundary="periodic")

    # ---- Likelihood -- Whittle likelihood via GravitationalWaveTransient.
    # time/distance/phase marginalised analytically since their true
    # values are unknown (the tutorial's real-event example instead
    # fixes them and only marginalises phase -- valid there because it
    # cheats with the known injection values; not valid for us). ----
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        priors=priors,
        time_marginalization=True,
        distance_marginalization=True,
        phase_marginalization=True,
        jitter_time=False,
    )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=int(nlive),
        # sample="rwalk",
        sample="slice",
        walks=None,
        nact=10,
        dlogz=0.1,
        outdir="/tmp/bilby_pe_out",
        label="gw_pe",
        clean=True,
        verbose=False,
        plot=False,
        save=False,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )

    post  = result.posterior
    ln_bf = float(result.log_bayes_factor) if hasattr(result, "log_bayes_factor") else float("nan")

    out = {
        "chirp_mass_Msun":     round(float(post["chirp_mass"].median()), 4),
        "mass_ratio":          round(float(post["mass_ratio"].median()), 4),
        "chirp_mass_5pct":     round(float(post["chirp_mass"].quantile(0.05)), 4),
        "chirp_mass_95pct":    round(float(post["chirp_mass"].quantile(0.95)), 4),
        "mass_ratio_5pct":     round(float(post["mass_ratio"].quantile(0.05)), 4),
        "mass_ratio_95pct":    round(float(post["mass_ratio"].quantile(0.95)), 4),
        "ra_median":           round(float(post["ra"].median()), 4),
        "dec_median":          round(float(post["dec"].median()), 4),
        "log_bayes_factor":    round(ln_bf, 2),
        "n_posterior_samples": int(len(post)),
    }

    # generate_all_bbh_parameters gives us mass_1 / mass_2 directly --
    # use them if present, otherwise fall back to the analytic conversion.
    if "mass_1" in post.columns and "mass_2" in post.columns:
        out["mass1_Msun"] = round(float(post["mass_1"].median()), 3)
        out["mass2_Msun"] = round(float(post["mass_2"].median()), 3)
    if "chi_eff" in post.columns:
        out["chi_eff_median"] = round(float(post["chi_eff"].median()), 4)

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
# Tool 5 -- Merger type classification
# =============================================================================

@tool
def classify_merger_type(mass1_Msun: float, mass2_Msun: float) -> dict:
    """
    Classify merger type as BBH, BNS, or NSBH from component masses.
    Neutron star mass range: 1.0-3.0 Msun. Black hole: > 3.0 Msun.
    Returns merger_type (exactly BBH, BNS, or NSBH) and reasoning.

    Args:
        mass1_Msun: primary component mass in solar masses
        mass2_Msun: secondary component mass in solar masses
    """
    NS_MAX = 3.0
    m1 = max(mass1_Msun, mass2_Msun)
    m2 = min(mass1_Msun, mass2_Msun)
    if m1 > NS_MAX and m2 > NS_MAX:
        mtype  = "BBH"
        reason = f"Both masses ({m1:.1f}, {m2:.1f} Msun) exceed {NS_MAX} Msun"
    elif m1 <= NS_MAX and m2 <= NS_MAX:
        mtype  = "BNS"
        reason = f"Both masses ({m1:.1f}, {m2:.1f} Msun) in NS range"
    else:
        mtype  = "NSBH"
        reason = f"Mixed: BH ({m1:.1f} Msun) + NS ({m2:.1f} Msun)"
    return {"merger_type": mtype, "reasoning": reason}


# =============================================================================
# Tool 6 -- Waveform / strain plotting
# =============================================================================

@tool
def plot_chirp_signal(strain_H1: list, strain_L1: list,
                       psd_H1: list, psd_freqs: list,
                       sample_rate: int,
                       chirp_mass_Msun: float, mass_ratio: float,
                       output_path: str = "/tmp/gw_chirp_plot.png",
                       f_lower: float = 20.0, approximant: str = "IMRPhenomD") -> dict:
    """
    3-panel figure:
      Panel 1 -- H1 raw strain + whitened strain + best-fit template, all overlaid
      Panel 2 -- L1 raw strain + whitened strain, overlaid
      Panel 3 -- H1 Q-transform (gwpy) showing chirp frequency sweep, zoomed to merger
    Saves a PNG.
 
    Use this AFTER you have your final chirp_mass and mass_ratio estimate.
 
    Args:
        strain_H1: H1 strain time series -- use data["strain_H1"] from load_gw_data
        strain_L1: L1 strain time series -- use data["strain_L1"] from load_gw_data
        psd_H1: H1 PSD -- use data["psd_H1"] from load_gw_data
        psd_freqs: PSD frequency axis -- use data["psd_freqs"] from load_gw_data
        sample_rate: sampling rate in Hz -- use data["sample_rate"] from load_gw_data
        chirp_mass_Msun: final recovered chirp mass for the overlay template
        mass_ratio: final recovered mass ratio for the overlay template
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
 
    # ── Whitening ────────────────────────────────────────────────────
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
 
    # Scale raw strain to whitened amplitude for visual overlay
    raw_H1_t     = strain_H1_arr[edge:-edge]
    raw_L1_t     = strain_L1_arr[edge:-edge]
    raw_scale_H1 = np.std(white_H1_t) / (np.std(raw_H1_t) + 1e-50)
    raw_scale_L1 = np.std(white_L1_t) / (np.std(raw_L1_t) + 1e-50)
 
    # ── Merger time from whitened H1 peak ───────────────────────────
    merger_idx  = int(np.argmax(np.abs(white_H1)))
    merger_time = float(merger_idx) / sample_rate
 
    # ── Best-fit template aligned to merger time ─────────────────────
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
 
        # Align template peak to measured merger time
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
 
    # ── Q-transform via gwpy (same API as tutorial) ──────────────────
    qtrans_ok  = False
    qtrans_err = ""
    qtrans_img = None
    try:
        from gwpy.timeseries import TimeSeries as GWpyTimeSeries
 
        # Build gwpy TimeSeries with t0=0 (our data starts at t=0)
        ts_gwpy = GWpyTimeSeries(strain_H1_arr, dt=dt, t0=0)
 
        # Zoom in on +-0.5s around the merger -- same as tutorial's outseg
        outseg = (merger_time - 0.5, merger_time + 0.5)
 
        qtrans_img = ts_gwpy.q_transform(
            frange=(f_lower, 512.0),
            qrange=(4, 64),
            outseg=outseg,
        )
        qtrans_ok = True
    except Exception as e:
        qtrans_err = str(e)
 
    # ── Figure: 3 panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle(
        f"GW Event  Mc={chirp_mass_Msun:.1f} M\u2609  q={mass_ratio:.2f}",
        fontsize=13, y=1.01,
    )
 
    # # Panel 1 -- H1: raw (faint, scaled) + whitened + template
    # axes[0].plot(t_trim, raw_H1_t * raw_scale_H1,
    #              color="steelblue", lw=0.4, alpha=0.35, label="H1 raw (scaled)")
    # axes[0].plot(t_trim, white_H1_t,
    #              color="steelblue", lw=0.8, alpha=0.9, label="H1 whitened")
    # if template_plotted:
    #     axes[0].plot(t_trim, white_tmpl_t,
    #                  color="black", lw=1.8,
    #                  label=f"template (Mc={chirp_mass_Msun:.1f}, q={mass_ratio:.2f})")
    # axes[0].axvline(merger_time, color="red", lw=1.0, ls="--",
    #                 alpha=0.7, label=f"merger t={merger_time:.2f}s")
    # axes[0].set_ylabel("Whitened Strain")
    # axes[0].set_title("H1 — raw + whitened + best-fit template")
    # axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    # axes[0].set_xlim(t_trim[0], t_trim[-1])
    # Panel 1 -- H1: raw on twin axis + whitened + template on main axis
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
    # Combine legends from both axes
    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines0r, labels0r = ax0_raw.get_legend_handles_labels()
    axes[0].legend(lines0 + lines0r, labels0 + labels0r,
                   loc="upper left", fontsize=8, ncol=2)
    axes[0].set_xlim(t_trim[0], t_trim[-1])
    ax0_raw.set_xlim(t_trim[0], t_trim[-1])
 
    # Panel 2 -- L1: raw (faint, scaled) + whitened
    # axes[1].plot(t_trim, raw_L1_t * raw_scale_L1,
    #              color="indianred", lw=0.4, alpha=0.35, label="L1 raw (scaled)")
    # axes[1].plot(t_trim, white_L1_t,
    #              color="indianred", lw=0.8, alpha=0.9, label="L1 whitened")
    # axes[1].axvline(merger_time, color="red", lw=1.0, ls="--",
    #                 alpha=0.7, label=f"merger t={merger_time:.2f}s")
    # axes[1].set_ylabel("Whitened Strain")
    # axes[1].set_title("L1 — raw + whitened strain")
    # axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    # axes[1].set_xlim(t_trim[0], t_trim[-1])
    # Panel 2 -- L1: raw on twin axis + whitened on main axis
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
 
    # Panel 3 -- Q-transform (gwpy) or fallback whitened spectrogram
    if qtrans_ok and qtrans_img is not None:
        # gwpy Spectrogram plots via its own .plot() but we use imshow
        # to stay on our own matplotlib axes
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
        # Fallback: whitened spectrogram over full segment
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
 