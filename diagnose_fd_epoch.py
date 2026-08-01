# """
# Isolate whether _colored_noise itself produces noise whose long-run
# spectrum matches the claimed PSD -- using scipy.signal.welch (a trusted,
# independently-implemented reference), completely bypassing any of my own
# hand-rolled formula. If scipy ALSO shows a large discrepancy, the bug is
# in _colored_noise itself, not in estimate_psd_welch.

# Run with the physics_agent_harness / GW_merger_bench venv:
#     python diagnose_colored_noise_scaling.py
# """
# import numpy as np
# from scipy.signal import welch
# from pycbc.psd import aLIGOZeroDetHighPower

# SAMPLE_RATE = 2048
# F_LOWER = 20.0
# DURATION = 128.0

# n_samples = int(DURATION * SAMPLE_RATE)
# flen = n_samples // 2 + 1
# delta_f = 1.0 / DURATION

# true_psd = np.array(aLIGOZeroDetHighPower(flen, delta_f, F_LOWER))
# true_freqs = np.linspace(0, SAMPLE_RATE / 2, flen)


# def colored_noise(psd_vals, psd_freqs, n_samples, sample_rate, seed):
#     """Exact copy of generate_dataset.py's _colored_noise."""
#     rng = np.random.default_rng(seed)
#     flen_local = n_samples // 2 + 1
#     freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
#     psd_interp = np.interp(freqs, psd_freqs, psd_vals, left=1e-40, right=1e-40)
#     psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
#     sigma_f = 0.5 * np.sqrt(psd_interp * sample_rate * n_samples)   # FIXED
#     noise_f = (rng.standard_normal(flen_local) + 1j * rng.standard_normal(flen_local)) * sigma_f
#     noise_f[0] = noise_f[0].real
#     noise_f[-1] = noise_f[-1].real
#     return np.fft.irfft(noise_f, n=n_samples).astype(np.float64)


# print("Generating noise from the TRUE PSD via _colored_noise...")
# noise = colored_noise(true_psd, true_freqs, n_samples, SAMPLE_RATE, seed=1234)

# print("Estimating PSD back via scipy.signal.welch (trusted reference implementation)...")
# f_scipy, psd_scipy = welch(
#     noise, fs=SAMPLE_RATE, nperseg=4 * SAMPLE_RATE, noverlap=2 * SAMPLE_RATE,
#     window="hann", scaling="density",
# )

# print("=" * 70)
# print(f"{'Freq (Hz)':>10}  {'True PSD':>14}  {'scipy welch PSD':>16}  {'Ratio':>10}")
# print("=" * 70)
# ratios = []
# for f in [30, 50, 100, 200, 500, 1000]:
#     i_true = np.argmin(np.abs(true_freqs - f))
#     i_sc = np.argmin(np.abs(f_scipy - f))
#     ratio = psd_scipy[i_sc] / true_psd[i_true]
#     ratios.append(ratio)
#     print(f"{f:10.1f}  {true_psd[i_true]:14.4e}  {psd_scipy[i_sc]:16.4e}  {ratio:10.6f}")

# print("=" * 70)
# mean_ratio = np.mean(ratios)
# print(f"Mean ratio (scipy/true): {mean_ratio:.6f}  (should be close to 1.0)")
# print("=" * 70)

# if 0.7 < mean_ratio < 1.4:
#     print("scipy agrees with the true PSD -- _colored_noise is correctly")
#     print("calibrated. The bug is isolated to estimate_psd_welch's own")
#     print("formula, not the noise generator.")
# else:
#     print("scipy ALSO disagrees significantly -- this means _colored_noise")
#     print("itself does not produce noise whose spectrum matches the stated")
#     print("PSD in the standard sense. This is a deeper, separate issue from")
#     print("estimate_psd_welch, worth investigating before trusting realism")
#     print("of any dataset's injected noise amplitude relative to its stated PSD.")


# """
# Generate synthetic colored noise from a KNOWN PSD (the same
# aLIGOZeroDetHighPower curve used elsewhere in this pipeline), then
# estimate the PSD back from that noise using estimate_psd_welch's exact
# method. Confirms the estimator's normalization convention is correct
# before it's trusted anywhere in the real pipeline.

# Run with the physics_agent_harness / GW_merger_bench venv:
#     python verify_psd_estimation.py
# """
# import numpy as np
# from pycbc.psd import aLIGOZeroDetHighPower

# SAMPLE_RATE = 2048
# F_LOWER = 20.0
# OFFSOURCE_DURATION = 128.0  # seconds, matching real off-source practice

# n_samples = int(OFFSOURCE_DURATION * SAMPLE_RATE)
# flen = n_samples // 2 + 1
# delta_f = 1.0 / OFFSOURCE_DURATION

# true_psd = np.array(aLIGOZeroDetHighPower(flen, delta_f, F_LOWER))
# true_freqs = np.linspace(0, SAMPLE_RATE / 2, flen)


# def make_colored_noise(psd_vals, psd_freqs, n_samples, sample_rate, seed):
#     """Same method as generate_dataset.py's _colored_noise, for consistency."""
#     rng = np.random.default_rng(seed)
#     flen_local = n_samples // 2 + 1
#     freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
#     psd_interp = np.interp(freqs, psd_freqs, psd_vals, left=1e-40, right=1e-40)
#     psd_interp = np.where(psd_interp > 0, psd_interp, 1e-40)
#     sigma_f = 0.5 * np.sqrt(psd_interp * sample_rate * n_samples)
#     noise_f = (rng.standard_normal(flen_local) + 1j * rng.standard_normal(flen_local)) * sigma_f
#     noise_f[0] = noise_f[0].real
#     noise_f[-1] = noise_f[-1].real
#     return np.fft.irfft(noise_f, n=n_samples).astype(np.float64)


# def estimate_psd_welch_local(strain_offsource, sample_rate, seg_duration=4.0, overlap_frac=0.5):
#     """Exact copy of estimate_psd_welch's logic, for standalone testing."""
#     strain_arr = np.asarray(strain_offsource, dtype=np.float64)
#     n_total = len(strain_arr)
#     seg_len = int(seg_duration * sample_rate)
#     step = max(1, int(seg_len * (1.0 - overlap_frac)))
#     window = np.hanning(seg_len)
#     window_norm = np.sum(window ** 2)
#     dt = 1.0 / sample_rate

#     periodograms = []
#     start = 0
#     while start + seg_len <= n_total:
#         segment = strain_arr[start:start + seg_len] * window
#         seg_fft = np.fft.rfft(segment)
#         periodogram = (2.0 / (sample_rate * window_norm)) * np.abs(seg_fft) ** 2
#         periodograms.append(periodogram)
#         start += step

#     periodograms = np.array(periodograms)
#     median_psd = np.median(periodograms, axis=0)
#     psd_estimate = median_psd / np.log(2.0)
#     freqs = np.fft.rfftfreq(seg_len, d=1.0 / sample_rate)
#     return psd_estimate, freqs, len(periodograms)


# print("Generating 128s of colored noise from the TRUE PSD...")
# noise = make_colored_noise(true_psd, true_freqs, n_samples, SAMPLE_RATE, seed=1234)

# print("Estimating PSD back from that noise via median-Welch...")
# psd_est, freqs_est, n_seg = estimate_psd_welch_local(noise, SAMPLE_RATE, seg_duration=4.0, overlap_frac=0.5)
# print(f"Used {n_seg} segments")

# # Compare true vs estimated PSD at several representative frequencies
# print("\n" + "=" * 70)
# print(f"{'Freq (Hz)':>10}  {'True PSD':>14}  {'Estimated PSD':>14}  {'Ratio':>8}")
# print("=" * 70)
# test_freqs = [30, 50, 100, 200, 500, 1000]
# all_ratios = []
# for f in test_freqs:
#     idx_true = np.argmin(np.abs(true_freqs - f))
#     idx_est = np.argmin(np.abs(freqs_est - f))
#     ratio = psd_est[idx_est] / true_psd[idx_true]
#     all_ratios.append(ratio)
#     print(f"{f:10.1f}  {true_psd[idx_true]:14.4e}  {psd_est[idx_est]:14.4e}  {ratio:8.3f}")

# mean_ratio = np.mean(all_ratios)
# print("=" * 70)
# print(f"Mean ratio (estimated/true): {mean_ratio:.3f}  (should be close to 1.0)")
# print("=" * 70)

# if 0.7 < mean_ratio < 1.4:
#     print("PASS -- normalization convention is correct (some scatter is expected")
#     print("and correct, given this is a genuine statistical estimate).")
# else:
#     print("FAIL -- systematic offset suggests a normalization bug. Check the")
#     print("periodogram scaling factor (2.0 / (sample_rate * window_norm)) and")
#     print("the median bias correction (1/ln(2)) before using this tool for real.")

"""
Directly measure the TRUE, effective SNR of an already-generated task's
strain data (using the exact same matched-filter formula as
generate_dataset.py's own _compute_snr), and compare it against the
network_snr value already recorded in that task's ground_truth.json.

If _colored_noise was miscalibrated (too quiet), the ACTUAL effective
SNR of the generated strain array will come out much higher than the
recorded "network_snr" label, since that label was computed independently
and correctly, but the noise floor added on top was not.

Usage:
    python check_real_snr_vs_labeled.py /path/to/task_dir
e.g.
    python check_real_snr_vs_labeled.py data/IMRPhenomD_zerospin/IMRPhenomD/000
"""
import sys
import json
import numpy as np

task_dir = sys.argv[1] if len(sys.argv) > 1 else "data/IMRPhenomD_zerospin/IMRPhenomD/000"

strain_H1 = np.load(f"{task_dir}/strain_H1.npy")
strain_L1 = np.load(f"{task_dir}/strain_L1.npy")
psd_H1 = np.load(f"{task_dir}/psd_H1.npy")
psd_L1 = np.load(f"{task_dir}/psd_L1.npy")
psd_freqs = np.load(f"{task_dir}/psd_freqs.npy")

with open(f"{task_dir}/ground_truth.json") as f:
    gt = json.load(f)

SAMPLE_RATE = gt.get("sample_rate", 2048)
F_LOWER = 20.0
dt = 1.0 / SAMPLE_RATE
n_samples = len(strain_H1)
delta_f = 1.0 / (n_samples * dt)


def compute_snr(sig, psd_vals, psd_freqs_arr):
    """Exact copy of generate_dataset.py's _compute_snr."""
    sig_f = np.fft.rfft(sig) * dt
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    psd_i = np.interp(freqs, psd_freqs_arr, psd_vals, left=1e-40, right=1e-40)
    psd_i = np.where(psd_i > 0, psd_i, 1e-40)
    integrand = np.abs(sig_f) ** 2 / psd_i
    mask = freqs >= F_LOWER
    snr_sq = 4.0 * np.sum(integrand[mask]) * delta_f
    return float(np.sqrt(max(snr_sq, 0.0)))


actual_snr_H1 = compute_snr(strain_H1, psd_H1, psd_freqs)
actual_snr_L1 = compute_snr(strain_L1, psd_L1, psd_freqs)
actual_network_snr = float(np.sqrt(actual_snr_H1 ** 2 + actual_snr_L1 ** 2))

print("=" * 70)
print(f"Task: {task_dir}")
print("=" * 70)
print(f"Labeled network_snr (ground_truth.json):  {gt['network_snr']:.3f}")
print(f"Labeled optimal_snr_H1/L1:                 {gt['optimal_snr_H1']:.3f} / {gt['optimal_snr_L1']:.3f}")
print("-" * 70)
print(f"ACTUAL measured SNR of strain_H1+strain_L1 (noise+signal, as saved):")
print(f"  H1: {actual_snr_H1:.3f}   L1: {actual_snr_L1:.3f}   network: {actual_network_snr:.3f}")
print("=" * 70)
ratio = actual_network_snr / gt["network_snr"]
print(f"Ratio (actual/labeled): {ratio:.3f}")
if ratio > 2.0:
    print("SIGNIFICANT MISMATCH -- the actual data is much louder (relative")
    print("to its own noise floor) than the labeled SNR suggests. Consistent")
    print("with _colored_noise under-generating noise amplitude.")
elif 0.7 < ratio < 1.4:
    print("Actual and labeled SNR are consistent -- no significant miscalibration.")
else:
    print("Some discrepancy present -- investigate further.")