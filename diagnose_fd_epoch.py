"""
Decisive test: is check_waveform_residual's persistent chi2_reduced~6
a real physical residual (from imperfect mass_ratio recovery) or a
remaining calibration bug in the tool?

Feed it the TRUE ground_truth.json parameters. If the tool is correctly
calibrated, chi2_reduced should land close to 1.0 for a genuine perfect
match. If it's still elevated, that's a tool bug, not a recovery issue.

Run with the physics_agent_harness venv:
    /home/sr/Desktop/code/physics_agent_harness/venv/bin/python test_chi2_with_truth.py
"""
import sys
sys.path.insert(0, "/home/sr/Desktop/code/physics_agent_harness")
import numpy as np
from tools.gw_tools import check_waveform_residual, load_gw_data

TASK_DIR = "/home/sr/Desktop/code/GW_merger_bench/data/IMRPhenomD_zerospin/IMRPhenomD/000"

# >>> UPDATE THESE from the CURRENT ground_truth.json <<<
TRUE_CHIRP_MASS =  5.9168
TRUE_MASS_RATIO = 0.2815  
TRUE_COA_TIME   = 9.9088

data = load_gw_data(
    strain_H1=f"{TASK_DIR}/strain_H1.npy",
    strain_L1=f"{TASK_DIR}/strain_L1.npy",
    psd_H1=f"{TASK_DIR}/psd_H1.npy",
    psd_L1=f"{TASK_DIR}/psd_L1.npy",
    psd_freqs=f"{TASK_DIR}/psd_freqs.npy",
    sample_rate=2048,
)

result_true = check_waveform_residual(
    strain_H1=data["strain_H1"], psd_H1=data["psd_H1"], psd_freqs=data["psd_freqs"],
    sample_rate=data["sample_rate"],
    chirp_mass_Msun=TRUE_CHIRP_MASS, mass_ratio=TRUE_MASS_RATIO,
    merger_time_s=TRUE_COA_TIME, f_lower=20.0, approximant="IMRPhenomD", n_bins=8,
)

print("=" * 70)
print("Using TRUE ground-truth parameters:")
print(f"  chirp_mass={TRUE_CHIRP_MASS}  mass_ratio={TRUE_MASS_RATIO}  t={TRUE_COA_TIME}")
print(result_true)
print("=" * 70)
if result_true["chi2_reduced"] < 2.0:
    print("chi2_reduced is near 1 for TRUE params -> the tool is correctly")
    print("calibrated. The ~6.3 seen with PE's recovered params is a REAL")
    print("physical residual from imperfect mass_ratio recovery, not a bug.")
else:
    print(f"chi2_reduced is STILL elevated ({result_true['chi2_reduced']}) even")
    print("with TRUE parameters -- this means check_waveform_residual itself")
    print("still has a remaining calibration bug, independent of recovery quality.")