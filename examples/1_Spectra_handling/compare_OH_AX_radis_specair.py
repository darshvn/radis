# -*- coding: utf-8 -*-
"""
===============================================================
Compare OH(A-X) electronic spectra: RADIS vs SpecAir
===============================================================

Auto-download the MoLLIST-OH line database from ExoMol and compute
the OH A-X electronic band (300-340 nm) under equilibrium and
non-equilibrium conditions. Compare with SpecAir reference data.

.. note::
    Electronic spectra computation in RADIS is a work in progress.
    RADIS currently shows ~10-20% discrepancy in integrated radiance
    compared to SpecAir, likely due to database differences.
    See :py:mod:`radis.test.validation.test_validation_vs_specair_OH_AX`
    for the full validation test suite.

"""

from os.path import join

import numpy as np
import pandas as pd

from radis import Spectrum, SpectrumFactory, plot_diff
from radis.test.utils import getValidationCase

# %% Load SpecAir reference
# --------------------------

csv_path = getValidationCase(
    join("test_validation_vs_specair_OH_AX_data", "radiance_spectrum.csv")
)

data = pd.read_csv(csv_path)
header = ",".join(data.columns)
w_ref = data.iloc[:, 0].values
I_ref = data.iloc[:, 1].values

mask = ~np.isnan(I_ref)
w_ref, I_ref = w_ref[mask], I_ref[mask]
idx = np.argsort(w_ref)
w_ref, I_ref = w_ref[idx], I_ref[idx]

# Convert W/cm^2/nm/sr -> mW/cm^2/sr/cm^-1
if "W_cm2_nm_sr" in header:
    I_ref = I_ref * w_ref**2 / 1e4

s_specair = Spectrum.from_array(
    w_ref, I_ref, "radiance", wunit="nm", unit="mW/cm2/sr/cm-1", name="SpecAir"
)

# %% Compute RADIS spectrum at equilibrium (400 K)
# --------------------------------------------------

sf = SpectrumFactory(
    wavelength_min=300,
    wavelength_max=340,
    molecule="OH",
    isotope="1",
    pressure=1,
    path_length=1,
    mole_fraction=1e-3,
    wstep=0.001,
    verbose=False,
    self_absorption=True,
)

sf.fetch_databank("exomol", "MoLLIST-OH", load_energies=True)

s_radis = sf.non_eq_spectrum(Ttrans=400, Trot=400, Tvib=400, Telec=10000, name="RADIS")
s_radis.apply_slit(0.1)  # nm, matching SpecAir spectral resolution

# %% Compare with plot_diff
# --------------------------

fig, [ax0, ax1] = plot_diff(
    s_radis,
    s_specair,
    "radiance",
    wunit="nm",
    Iunit="mW/cm2/sr/cm-1",
    method="diff",
)

ax0.set_title("OH(A-X) equilibrium at 400 K: RADIS vs SpecAir")

# Print integrated radiance ratio (filter NaN from slit convolution edges, sort by wavelength)
w_r, I_r = s_radis.get("radiance", wunit="nm", Iunit="mW/cm2/sr/cm-1")
mask_r = ~np.isnan(I_r)
w_r, I_r = w_r[mask_r], I_r[mask_r]
idx_r = np.argsort(w_r)
w_r, I_r = w_r[idx_r], I_r[idx_r]

w_s, I_s = s_specair.get("radiance", wunit="nm", Iunit="mW/cm2/sr/cm-1")
mask_s = ~np.isnan(I_s)
w_s, I_s = w_s[mask_s], I_s[mask_s]
idx_s = np.argsort(w_s)
w_s, I_s = w_s[idx_s], I_s[idx_s]

ratio = np.trapezoid(I_r, w_r) / np.trapezoid(I_s, w_s)
print(f"Integrated radiance ratio (RADIS/SpecAir): {ratio:.2f}")
