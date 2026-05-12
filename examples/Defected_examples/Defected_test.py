import numpy as np

from phonon_lifetime.Normal_Mode_New import (
    Plot_displacement,
    System,
    calculate_normal_modes,
)

# Defected System
# system = System(
#     element="Ni",
#     primitive_cell=np.diag([1.0, 1.0, 1.0]),
#     spring_constant=(1.0, 1.0, 0.0),
#     n_repeats=(15, 1, 1),
#     vacancy=(3, 0, 0),
# )
# results = calculate_normal_modes(system)
# Omega1 = results.omega
# Omega1 = Omega1[0, 28:]
# print(Omega1)
# Pristine
system = System(
    element="Ni",
    primitive_cell=np.diag([1.0, 1.0, 1.0]),
    spring_constant=(1.0, 1.0, 0.0),
    n_repeats=(15, 1, 1),
    vacancy=None,
)
results = calculate_normal_modes(system)
branch = 2
q = (3 / 15, 0, 0)
fig = Plot_displacement(results, 0, branch, q, vacancy=None)
# Save figure
out_path = f"./examples/Defected_examples/Defected_Results/1D_Displacement_branch_all_q{q[0]:.3f}_{q[1]:.3f}_{q[2]:.3f}.png"
fig.savefig(out_path)
print(out_path)
# Omega1, Omega2 shape = (15, 1)
# Omega1_sorted = np.sort(Omega1, axis=0)
# pos = 10
# Omega1_sorted = np.insert(Omega1_sorted, pos, 0)
# Omega1_sorted = Omega1_sorted[1::]
# Omega2_sorted = np.sort(Omega2, axis=0)[1::]
# print(Omega1_sorted)
# print(Omega2_sorted)
# diff = Omega1_sorted - Omega2_sorted
# x = np.arange(diff.shape[0])
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(x, diff / Omega2_sorted, marker="o")
# ax.set_xlabel("Index")
# ax.set_ylabel("Omega1 - Omega2")
# ax.set_title("Difference between Omega1 and Omega2")
# ax.grid(True, alpha=0.3)

# fig.savefig("./examples/Defected_examples/Defected_Results/Frequencies_Diff.png")
# print("./examples/Defected_examples/Defected_Results/Frequencies_Diff.png")


# Pristine system
# system = System(
#     element="Ni",
#     primitive_cell=np.diag([1.0, 1.0, 1.0]),
#     spring_constant=(1.0, 1.0, 0.0),
#     n_repeats=(15, 1, 1),
#     vacancy=None,
# )
# results = calculate_normal_modes(system)
# q_vals = results.q_vals
# print(q_vals[:, 0])

# fig = Plot_dispersion(results=results, branch=2)
# out_path = "./examples/Defected_examples/Pristine_Results/Dispersion_of_15.png"
# print(out_path)
# fig.savefig(out_path, dpi=300)
# branch = 2
# q = (1 / 24, 0, 0)
# t = 0
# fig = Plot_displacement(results, t, branch, q)
# Save figure
# out_path = f"./examples/Defected_examples/Pristine_Results/1D_Displacement_branch{branch}_q{q[0]:.3f}_{q[1]:.3f}_{q[2]:.3f}.png"
# print(out_path)
# fig.savefig(out_path, dpi=300)
