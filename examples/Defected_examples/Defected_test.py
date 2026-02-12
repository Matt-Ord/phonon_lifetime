import numpy as np

from phonon_lifetime.Normal_Mode_New import (
    Plot_displacement,
    System,
    calculate_normal_modes,
)

# Defected System
system = System(
    element="Ni",
    primitive_cell=np.diag([1.0, 1.0, 1.0]),
    spring_constant=(1.0, 1.0, 0.0),
    n_repeats=(3, 3, 1),
    vacancy=(1, 1, 0),
)
results = calculate_normal_modes(system)
Frequencies = results.omega

branch = 18
q = (0, 0, 0)
t = 0
fig = Plot_displacement(results, t, branch, q, vacancy=(1, 1, 0))
# Save figure
out_path = f"./examples/Defected_examples/Defected_Results/Displacement_branch{branch}_q{q[0]:.3f}_{q[1]:.3f}_{q[2]:.3f}.png"
print(out_path)
fig.savefig(out_path, dpi=300)

# Pristine system
# system = System(
#     element="Ni",
#     primitive_cell=np.diag([1.0, 1.0, 1.0]),
#     spring_constant=(1.0, 1.0, 0.0),
#     n_repeats=(3, 3, 1),
#     vacancy=None,
# )
# results = calculate_normal_modes(system)
# branch = 2
# q = (0, 2 / 3, 0)
# t = 0
# fig = Plot_displacement(results, t, branch, q)
# # Save figure
# out_path = f"./examples/Defected_examples/Pristine_Results/Displacement_branch{branch}_q{q[0]:.3f}_{q[1]:.3f}_{q[2]:.3f}.png"
# print(out_path)
# fig.savefig(out_path, dpi=300)
