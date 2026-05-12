from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from phonon_lifetime.Mass_Defect_Phonon import (
    Plot_displacement,
    Plot_overlap,
    Plot_scattering_rate,
    System,
    calculate_normal_modes,
    fermi_golden_rule2,
)

defect_cases = [
    # [((2, 2, 0), 48.0)],
    # [((2, 2, 0), 53.0)],
    [((2, 2, 0), 45.0)],
]
# Common lattice/system parameters shared by p/d system
prim = np.diag([1.0, 1.0, 1.0])
k = (1.0, 1.0, 1.0)
nrep = (11, 11, 1)

# Compute pristine system
sys_pri = System(
    element="Ni",
    primitive_cell=prim,
    spring_constant=k,
    defects=None,
    n_repeats=nrep,
)

res_pri = calculate_normal_modes(sys_pri)
sys_unit = System(
    element="Ni",
    primitive_cell=prim,
    spring_constant=k,
    defects=None,
    n_repeats=nrep,
)
# Compute and plot scattering rate curves for different defects
defect = [((5, 5, 0), 8.0)]
sys_def = System(
    element="Ni",
    primitive_cell=prim,
    spring_constant=k,
    defects=defect,
    n_repeats=nrep,
)

fig, ax = plt.subplots(figsize=(6, 4))
res_def = calculate_normal_modes(sys_def)
omega = res_pri.omega

band_q_list = [
    # [[1 / 5, 1 / 5, 0], 1],
    [[2 / 11, 5 / 11, 0], 2],
    # [[1 / 5, 1 / 5, 0], 2],
]

for q, band_sel in band_q_list:
    rate = fermi_golden_rule2(res_pri, res_def, band_sel, q, sys_unit)
    Omegas = res_pri.omega.reshape(-1)

    fig = Plot_scattering_rate(fig, ax, rate, res_pri, q)

    indices = np.count_nonzero(np.isclose(rate, 0, atol=1e-4))
    # print(rate.size - indices, "non_zero scattering")
    Plot_overlap(res_pri, res_def, band_sel, q)
outname = f"./examples/Lifetime_Computation/Lifetime_results/scattering_rate_band{band_sel}.png"
fig.savefig(outname, dpi=250, bbox_inches="tight")
plt.close(fig)
print("Saved:", outname)


mode_i = res_def.get_mode(181, (0, 0, 0))
Plot_displacement(mode_i, 0)
# print(res_pri.q_vals)
# plot overlap between selected pristine state and defected states
# band = 2
# q = (0, 1 / 5, 0)
# Plot_overlap(res_pri, res_def, band, q)
