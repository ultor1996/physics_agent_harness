"""
Unit tests for gw_tools.py

Tests each tool against a synthetic BBH injection with known parameters.
Run from the physics_agent_harness directory:
    python tests/test_gw_tools.py

Each test prints PASS/FAIL with the actual vs expected values.
"""

import sys
import os
import numpy as np
import tempfile

# ── Synthetic injection parameters (ground truth) ───────────────────
TRUE_MC       = 28.0      # chirp mass Msun
TRUE_Q        = 0.75      # mass ratio
TRUE_SPIN1Z   = 0.0
TRUE_SPIN2Z   = 0.0
TRUE_INC      = 0.0       # face-on
TRUE_DISTANCE = 500.0     # Mpc
SAMPLE_RATE   = 2048
SEGMENT_DUR   = 16
F_LOWER       = 20.0
APPROXIMANT   = "IMRPhenomD"
COA_OFFSET    = SEGMENT_DUR * 0.67   # merger at 67% of segment

# ── Tolerances ──────────────────────────────────────────────────────
MC_TOL_FRAC   = 0.05   # 5% chirp mass
Q_TOL_ABS     = 0.15   # 0.15 absolute mass ratio
SNR_MIN       = 5.0    # matched filter should find SNR > 5

# ── Colour printing ─────────────────────────────────────────────────
def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

results = []

def report(name, passed, detail=""):
    tag = green("PASS") if passed else red("FAIL")
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, passed))


# ════════════════════════════════════════════════════════════════════
# Generate synthetic data
# ════════════════════════════════════════════════════════════════════
def make_synthetic_data():
    """
    Generate a synthetic BBH injection with known parameters.
    Returns arrays ready to pass into load_gw_data's output format.
    """
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries

    # Derived masses
    q  = TRUE_Q
    Mc = TRUE_MC
    m1 = Mc * (1.0 + q)**(1.0/5.0) / q**(3.0/5.0)
    m2 = q * m1

    dt    = 1.0 / SAMPLE_RATE
    flen  = int(SEGMENT_DUR * SAMPLE_RATE / 2) + 1
    n     = int(SEGMENT_DUR * SAMPLE_RATE)
    delta_f = 1.0 / SEGMENT_DUR

    # Waveform
    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=m1, mass2=m2,
        spin1z=TRUE_SPIN1Z, spin2z=TRUE_SPIN2Z,
        inclination=TRUE_INC, coa_phase=0.0,
        delta_t=dt, f_lower=F_LOWER, distance=TRUE_DISTANCE,
    )

    # PSD
    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)
    psd_L1 = aLIGOZeroDetHighPower(flen, delta_f, F_LOWER)

    # Antenna pattern
    ra, dec, psi = 1.57, 0.0, 0.0
    ref_gps = 1264316116.0
    gps_coa = ref_gps + COA_OFFSET
    det_H1  = Detector("H1")
    det_L1  = Detector("L1")
    fp_H1, fc_H1 = det_H1.antenna_pattern(ra, dec, psi, gps_coa)
    fp_L1, fc_L1 = det_L1.antenna_pattern(ra, dec, psi, gps_coa)

    # Project signal
    def project(hp, hc, fp, fc):
        sig     = fp * hp + fc * hc
        arr     = np.zeros(n)
        idx     = int(COA_OFFSET * SAMPLE_RATE)
        end     = min(idx, len(sig))
        arr[idx - end: idx] = np.array(sig)[len(sig) - end:]
        return arr

    sig_H1 = project(hp, hc, fp_H1, fc_H1)
    sig_L1 = project(hp, hc, fp_L1, fc_L1)

    # Coloured noise
    psd_freqs = np.linspace(0, SAMPLE_RATE / 2, flen)
    psd_vals  = np.array(psd_H1)

    def colored_noise(seed):
        rng    = np.random.default_rng(seed)
        freqs  = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
        psd_i  = np.interp(freqs, psd_freqs, psd_vals, left=1e-40, right=1e-40)
        psd_i  = np.where(psd_i > 0, psd_i, 1e-40)
        sf     = np.sqrt(psd_i * SAMPLE_RATE / 2)
        nf     = (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) * sf
        nf[0]  = nf[0].real; nf[-1] = nf[-1].real
        return np.fft.irfft(nf, n=n).astype(np.float64)

    strain_H1 = colored_noise(42)  + sig_H1
    strain_L1 = colored_noise(123) + sig_L1

    return {
        "strain_H1":   strain_H1,
        "strain_L1":   strain_L1,
        "psd_H1":      psd_vals,
        "psd_L1":      psd_vals,
        "psd_freqs":   psd_freqs,
        "sample_rate": SAMPLE_RATE,
        "delta_t":     dt,
        "delta_f":     float(psd_freqs[1] - psd_freqs[0]),
        "duration":    float(SEGMENT_DUR),
        "n_samples":   n,
    }


# ════════════════════════════════════════════════════════════════════
# Test 0 — imports
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 0: Imports ==="))
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tools.gw_tools import (
        load_gw_data,
        seed_pe_prior_via_matched_filter,
        run_bayesian_pe,
        classify_merger_type,
        plot_chirp_signal,
    )
    report("import gw_tools", True)
except Exception as e:
    report("import gw_tools", False, str(e))
    print(red("Cannot continue without gw_tools — exiting"))
    sys.exit(1)

try:
    import pycbc
    report("import pycbc", True)
except Exception as e:
    report("import pycbc", False, str(e))

try:
    import bilby
    report("import bilby", True)
except Exception as e:
    report("import bilby", False, str(e))


# ════════════════════════════════════════════════════════════════════
# Generate synthetic data once, reuse across all tests
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Generating synthetic injection ==="))
print(f"  True Mc={TRUE_MC} Msun  q={TRUE_Q}  "
      f"spin1z={TRUE_SPIN1Z}  distance={TRUE_DISTANCE} Mpc")
try:
    data = make_synthetic_data()
    print(f"  Strain H1 shape: {data['strain_H1'].shape}  "
          f"peak={np.max(np.abs(data['strain_H1'])):.2e}")
    print(green("  Synthetic data generated OK"))
except Exception as e:
    print(red(f"  ERROR generating synthetic data: {e}"))
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════
# Test 1 — load_gw_data
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 1: load_gw_data ==="))
with tempfile.TemporaryDirectory() as tmpdir:
    # Save arrays to .npy files
    for key in ["strain_H1", "strain_L1", "psd_H1", "psd_L1", "psd_freqs"]:
        np.save(os.path.join(tmpdir, f"{key}.npy"), data[key])

    try:
        loaded = load_gw_data(
            strain_H1 = os.path.join(tmpdir, "strain_H1.npy"),
            strain_L1 = os.path.join(tmpdir, "strain_L1.npy"),
            psd_H1    = os.path.join(tmpdir, "psd_H1.npy"),
            psd_L1    = os.path.join(tmpdir, "psd_L1.npy"),
            psd_freqs = os.path.join(tmpdir, "psd_freqs.npy"),
            sample_rate = SAMPLE_RATE,
        )

        n_expected = int(SEGMENT_DUR * SAMPLE_RATE)
        report("returns dict",
               isinstance(loaded, dict),
               f"type={type(loaded)}")
        report("strain_H1 shape",
               len(loaded["strain_H1"]) == n_expected,
               f"got {len(loaded['strain_H1'])} expected {n_expected}")
        report("sample_rate correct",
               loaded["sample_rate"] == SAMPLE_RATE,
               f"got {loaded['sample_rate']}")
        report("duration correct",
               abs(loaded["duration"] - SEGMENT_DUR) < 0.01,
               f"got {loaded['duration']:.2f}s expected {SEGMENT_DUR}s")
        report("all keys present",
               all(k in loaded for k in
                   ["strain_H1","strain_L1","psd_H1","psd_L1",
                    "psd_freqs","sample_rate","duration","delta_t","delta_f"]),
               )
    except Exception as e:
        report("load_gw_data runs without error", False, str(e))


# ════════════════════════════════════════════════════════════════════
# Test 2 — seed_pe_prior_via_matched_filter
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 2: seed_pe_prior_via_matched_filter ==="))
mf = None
try:
    mf = seed_pe_prior_via_matched_filter(
        strain      = data["strain_H1"].tolist(),
        psd         = data["psd_H1"].tolist(),
        psd_freqs   = data["psd_freqs"].tolist(),
        strain_L1   = data["strain_L1"].tolist(),
        psd_L1      = data["psd_L1"].tolist(),
        sample_rate = SAMPLE_RATE,
        approximant = APPROXIMANT,
        f_lower     = F_LOWER,
    )

    required_keys = ["best_chirp_mass_Msun","best_mass_ratio",
                     "best_snr","best_mass1","best_mass2","merger_time_s"]
    report("returns all keys",
           all(k in mf for k in required_keys),
           f"got keys: {list(mf.keys())}")

    mc_err = abs(mf["best_chirp_mass_Msun"] - TRUE_MC) / TRUE_MC
    report("Mc within 20% of true",
           mc_err < 0.20,
           f"MF Mc={mf['best_chirp_mass_Msun']:.2f} true={TRUE_MC:.2f} err={mc_err*100:.1f}%")

    report("SNR > minimum threshold",
           mf["best_snr"] > SNR_MIN,
           f"SNR={mf['best_snr']:.1f} (min={SNR_MIN})")

    report("merger_time_s reasonable",
           0 < mf["merger_time_s"] < SEGMENT_DUR,
           f"merger_time_s={mf['merger_time_s']:.2f}s "
           f"(expected ~{COA_OFFSET:.2f}s)")

    mt_direct  = abs(mf["merger_time_s"] - COA_OFFSET)
    mt_wrapped = abs(mf["merger_time_s"] - (SEGMENT_DUR - COA_OFFSET))
    mt_err     = min(mt_direct, mt_wrapped)
    report("merger_time_s within 1s of true (SNR wraparound accounted for)",
           mt_err < 1.0,
           f"direct_err={mt_direct:.3f}s  wrapped_err={mt_wrapped:.3f}s")

    print(f"  MF result: Mc={mf['best_chirp_mass_Msun']:.2f}  "
          f"q={mf['best_mass_ratio']:.3f}  SNR={mf['best_snr']:.1f}  "
          f"t_merger={mf['merger_time_s']:.2f}s")

except Exception as e:
    report("seed_pe_prior_via_matched_filter runs", False, str(e))


# ════════════════════════════════════════════════════════════════════
# Test 3 — classify_merger_type
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 3: classify_merger_type ==="))
test_cases = [
    (35.0, 28.0, "BBH"),    # two BHs
    (1.4,  1.2,  "BNS"),    # two NSs
    (8.0,  1.4,  "NSBH"),   # BH + NS
    (30.0, 0.5,  "NSBH"),   # BH + NS edge case
]
for m1, m2, expected in test_cases:
    try:
        result = classify_merger_type(mass1_Msun=m1, mass2_Msun=m2)
        ok = result["merger_type"] == expected
        report(f"classify ({m1},{m2}) → {expected}",
               ok,
               f"got {result['merger_type']}")
    except Exception as e:
        report(f"classify ({m1},{m2}) → {expected}", False, str(e))


# ════════════════════════════════════════════════════════════════════
# Test 4 — run_bayesian_pe  (quick sanity check, not full accuracy)
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 4: run_bayesian_pe (quick, nlive=50) ==="))
print("  (using nlive=50 for speed — not for accuracy assessment)")
pe = None
try:
    merger_time = mf["merger_time_s"] if mf else COA_OFFSET
    mc_guess    = mf["best_chirp_mass_Msun"] if mf else TRUE_MC

    pe = run_bayesian_pe(
        strain_H1        = data["strain_H1"].tolist(),
        psd_H1           = data["psd_H1"].tolist(),
        strain_L1        = data["strain_L1"].tolist(),
        psd_L1           = data["psd_L1"].tolist(),
        psd_freqs        = data["psd_freqs"].tolist(),
        sample_rate      = SAMPLE_RATE,
        chirp_mass_guess = mc_guess,
        mass_ratio_guess = mf["best_mass_ratio"] if mf else TRUE_Q,
        merger_time_s    = merger_time,
        f_lower          = F_LOWER,
        approximant      = APPROXIMANT,
        nlive            = 50,   # fast — just checking it runs
    )

    required_pe_keys = ["chirp_mass_Msun","mass_ratio","mass1_Msun","mass2_Msun",
                        "log_bayes_factor","n_posterior_samples"]
    report("returns all required keys",
           all(k in pe for k in required_pe_keys),
           f"got: {list(pe.keys())}")

    report("chirp_mass_Msun is finite positive",
           np.isfinite(pe["chirp_mass_Msun"]) and pe["chirp_mass_Msun"] > 0,
           f"got {pe['chirp_mass_Msun']:.3f}")

    report("mass_ratio in (0,1]",
           0 < pe["mass_ratio"] <= 1.0,
           f"got {pe['mass_ratio']:.3f}")

    report("mass1 > mass2",
           pe["mass1_Msun"] >= pe["mass2_Msun"],
           f"m1={pe['mass1_Msun']:.2f}  m2={pe['mass2_Msun']:.2f}")

    report("n_posterior_samples > 0",
           pe["n_posterior_samples"] > 0,
           f"got {pe['n_posterior_samples']}")

    mc_err_pe = abs(pe["chirp_mass_Msun"] - TRUE_MC) / TRUE_MC
    report("Mc within 20% (nlive=50 rough check)",
           mc_err_pe < 0.20,
           f"PE Mc={pe['chirp_mass_Msun']:.2f} true={TRUE_MC:.2f} err={mc_err_pe*100:.1f}%")

    print(f"  PE result: Mc={pe['chirp_mass_Msun']:.2f}  "
          f"q={pe['mass_ratio']:.3f}  "
          f"lnBF={pe['log_bayes_factor']:.1f}  "
          f"N_post={pe['n_posterior_samples']}")

except Exception as e:
    report("run_bayesian_pe runs without error", False, str(e))
    import traceback; traceback.print_exc()


# ════════════════════════════════════════════════════════════════════
# Test 5 — plot_chirp_signal
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Test 5: plot_chirp_signal ==="))
with tempfile.TemporaryDirectory() as tmpdir:
    out_path = os.path.join(tmpdir, "test_chirp.png")
    try:
        plot_result = plot_chirp_signal(
            strain_H1       = data["strain_H1"].tolist(),
            strain_L1       = data["strain_L1"].tolist(),
            psd_H1          = data["psd_H1"].tolist(),
            psd_freqs       = data["psd_freqs"].tolist(),
            sample_rate     = SAMPLE_RATE,
            chirp_mass_Msun = pe["chirp_mass_Msun"] if pe else TRUE_MC,
            mass_ratio      = pe["mass_ratio"] if pe else TRUE_Q,
            output_path     = out_path,
            f_lower         = F_LOWER,
            approximant     = APPROXIMANT,
        )

        report("runs without error", True)
        report("output file created",
               os.path.exists(out_path),
               f"path={out_path}")
        if os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            report("output file non-empty",
                   size_kb > 10,
                   f"size={size_kb:.1f} KB")
        report("returns plot_path key",
               "plot_path" in plot_result,
               f"keys={list(plot_result.keys())}")
        report("template_overlaid reported",
               "template_overlaid" in plot_result)
        report("qtransform_used reported",
               "qtransform_used" in plot_result,
               f"qtransform_used={plot_result.get('qtransform_used')}")

    except Exception as e:
        report("plot_chirp_signal runs without error", False, str(e))
        import traceback; traceback.print_exc()


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
print(bold("\n=== Summary ==="))
n_pass = sum(1 for _, p in results if p)
n_fail = sum(1 for _, p in results if not p)
print(f"  {green(str(n_pass))} passed   {red(str(n_fail))} failed   "
      f"({len(results)} total)")

if n_fail > 0:
    print(red("\n  Failed tests:"))
    for name, passed in results:
        if not passed:
            print(f"    ✗ {name}")

print()
sys.exit(0 if n_fail == 0 else 1)