from smolagents import tool
import numpy as np


@tool
def load_gw_data(strain_H1: str, strain_L1: str,
                 psd_H1: str, psd_L1: str,
                 psd_freqs: str, times: str) -> dict:
    """
    Load gravitational-wave strain and PSD data from .npy files.
    Always call this first before any other tool.
    Returns strain_H1, strain_L1, psd_H1, psd_L1, psd_freqs, times,
    sample_rate, duration, delta_t, delta_f as a dict.

    Args:
        strain_H1: absolute path to H1 strain .npy file
        strain_L1: absolute path to L1 strain .npy file
        psd_H1: absolute path to H1 PSD .npy file
        psd_L1: absolute path to L1 PSD .npy file
        psd_freqs: absolute path to PSD frequency axis .npy file
        times: absolute path to times array .npy file
    """
    strain_H1_arr = np.load(strain_H1)
    strain_L1_arr = np.load(strain_L1)
    psd_H1_arr    = np.load(psd_H1)
    psd_L1_arr    = np.load(psd_L1)
    psd_freqs_arr = np.load(psd_freqs)
    times_arr     = np.load(times)

    delta_t     = float(times_arr[1] - times_arr[0])
    sample_rate = int(round(1.0 / delta_t))
    duration    = float(times_arr[-1] - times_arr[0])
    delta_f     = float(psd_freqs_arr[1] - psd_freqs_arr[0]) if len(psd_freqs_arr) > 1 else 1.0 / duration

    return {
        "strain_H1":   strain_H1_arr,
        "strain_L1":   strain_L1_arr,
        "psd_H1":      psd_H1_arr,
        "psd_L1":      psd_L1_arr,
        "psd_freqs":   psd_freqs_arr,
        "times":       times_arr,
        "sample_rate": sample_rate,
        "delta_t":     delta_t,
        "delta_f":     delta_f,
        "duration":    duration,
        "n_samples":   len(strain_H1_arr),
    }


@tool
def matched_filter_chirp_mass(strain: list, psd: list, psd_freqs: list,
                               sample_rate: int, approximant: str = "IMRPhenomD",
                               f_lower: float = 20.0) -> dict:
    """
    Estimate chirp mass by running a PyCBC matched filter bank over a
    grid of chirp masses and returning the one with the highest SNR.
    Searches chirp masses from 5 to 150 Msun with equal-mass assumption.
    Returns best_chirp_mass_Msun, best_snr, best_mass1, best_mass2,
    and snr_vs_chirp_mass list of [Mc, snr] pairs.

    Args:
        strain: H1 strain time series as array or list
        psd: H1 power spectral density as array or list
        psd_freqs: PSD frequency axis as array or list
        sample_rate: sampling rate in Hz e.g. 2048
        approximant: waveform approximant string e.g. IMRPhenomD
        f_lower: lower frequency cutoff in Hz e.g. 20.0
    """
    from pycbc.types import TimeSeries, FrequencySeries
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.waveform import get_fd_waveform
    from pycbc.filter import matched_filter, sigma

    delta_t   = 1.0 / sample_rate
    strain_ts = TimeSeries(np.array(strain, dtype=np.float64), delta_t=delta_t)
    N         = len(strain_ts)
    delta_f   = 1.0 / strain_ts.duration
    flen      = N // 2 + 1

    psd_arr   = np.array(psd, dtype=np.float64)
    freqs_arr = np.array(psd_freqs, dtype=np.float64)
    psd_fs    = FrequencySeries(psd_arr, delta_f=float(freqs_arr[1] - freqs_arr[0]))
    psd_interp = interpolate(psd_fs, delta_f)
    psd_trunc  = inverse_spectrum_truncation(
        psd_interp, int(4 * sample_rate), low_frequency_cutoff=f_lower
    )

    chirp_masses = np.logspace(np.log10(5), np.log10(150), 30)
    results      = []

    for Mc in chirp_masses:
        m1 = float(Mc * 2**(1.0/5.0))
        m2 = m1
        try:
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

            snr_ts   = matched_filter(hp, strain_ts.to_frequencyseries(),
                                      psd=psd_trunc, low_frequency_cutoff=f_lower)
            peak_snr = float(abs(snr_ts).numpy().max())
            results.append((float(Mc), peak_snr, m1, m2))
        except Exception:
            continue

    if not results:
        return {"best_chirp_mass_Msun": 25.0, "best_snr": 0.0,
                "best_mass1": 29.0, "best_mass2": 29.0,
                "snr_vs_chirp_mass": []}

    best = max(results, key=lambda x: x[1])
    return {
        "best_chirp_mass_Msun": best[0],
        "best_snr":             best[1],
        "best_mass1":           best[2],
        "best_mass2":           best[3],
        "snr_vs_chirp_mass":    [[r[0], r[1]] for r in results],
    }


@tool
def estimate_component_masses(chirp_mass_Msun: float,
                               mass_ratio_guess: float = 0.8) -> dict:
    """
    Convert chirp mass and mass ratio into component masses m1 and m2.
    mass_ratio q = m2/m1 where m1 >= m2, so q is in (0, 1].
    Returns mass1_Msun, mass2_Msun, total_mass_Msun, mass_ratio.

    Args:
        chirp_mass_Msun: chirp mass in solar masses
        mass_ratio_guess: mass ratio q = m2/m1, start with 0.8 and try 0.5 if SNR is low
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