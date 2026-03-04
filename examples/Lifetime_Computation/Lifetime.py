from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.Mass_Defect_Phonon import (
    Plot_overlap,
    Plot_scattering_rate,
    System,
    calculate_normal_modes,
    fermi_golden_rule2,
)

defect_cases = [
    [((2, 2, 0), 48.0)],
    [((2, 2, 0), 53.0)],
    [((2, 2, 0), 58.0)],
]
# Common lattice/system parameters shared by p/d system
prim = np.diag([1.0, 1.0, 1.0])
k = (1.0, 1.0, 1.0)
nrep = (5, 3, 1)

# Compute pristine system
sys_pri = System(
    element="Ni",
    primitive_cell=prim,
    spring_constant=k,
    defects=None,
    n_repeats=nrep,
)
res_pri = calculate_normal_modes(sys_pri)

# Compute and plot scattering rate curves for different defects
band_sel = 2
fig, ax = plt.subplots(figsize=(6, 4))
for defect in defect_cases:
    sys_def = System(
        element="Ni",
        primitive_cell=prim,
        spring_constant=k,
        defects=defect,
        n_repeats=nrep,
    )
    res_def = calculate_normal_modes(sys_def)
    rate = fermi_golden_rule2(res_pri, res_def, band_sel, (2 / 5, 1 / 3, 0))
    print(rate.shape)
    fig = Plot_scattering_rate(fig, ax, rate, res_pri, defect)


outname = f"./examples/Lifetime_Computation/Lifetime_results/scattering_rate_band_{band_sel}.png"
fig.savefig(outname, dpi=250, bbox_inches="tight")
plt.close(fig)
print("Saved:", outname)
# plot overlap between selected pristine state and defected states
band = 2
q = (0, 1 / 3, 0)
Plot_overlap(res_pri, res_def, band, q)
