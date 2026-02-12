import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.Normal_Mode_New import (
    System,
    build_force_constant_matrix,
    calculate_normal_modes,
)


def Plot_displacement(
    result: NormalModeResult,
    time: float = 0,
    branch: int = 0,
    q: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    mode = result.get_mode(branch, q)
    u = mode.get_displacement()
    X, Y = mode.system.get_atom_centres()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.quiver(X, Y, u[:, 0], u[:, 1], angles="xy", scale_units="xy", scale=1.0)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out_path = f"./examples/2D_Results/Eigenvectors/Displacement_branch{branch}_q{q[0]:.3f}_{q[1]:.3f}_{q[2]:.3f}.png"
    print(out_path)
    fig.savefig(out_path, dpi=300)


system = System(
    element="Ni",
    primitive_cell=np.diag([1.0, 1.0, 1.0]),
    spring_constant=(1.0, 1.0, 0.0),
    Defected_cell_size=(3, 3, 1),
    vacancy=(1, 1, 0),
)
fc = build_force_constant_matrix(system)
results = calculate_normal_modes(system)
mode = results.get_mode(5, [0, 0, 0])
u = mode.get_displacement()
print(u.shape)
Plot_displacement(results, 0, 2, (0, 0, 0))
