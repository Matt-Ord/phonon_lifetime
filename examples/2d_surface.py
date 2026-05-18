import matplotlib.pyplot as plt

from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.phonon import (
    CubicPoint,
    DispersionPath,
    animate_phonon_xy,
    animate_phonon_xyz,
    as_gamma_phonon,
    get_dispersion_path,
    get_mesh_phonons,
    plot_dispersion_path,
    plot_phonon_xy,
)
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, True, False), threshold=(0.0, 1.1)
    )
    result = get_mesh_phonons(system, n_repeats=(11, 11, 1))

    phonon = result.select_phonon(branch=2, iq=(1, 0, 0))
    phonon = as_gamma_phonon(phonon)
    fig, ax, _ = plot_phonon_xy(phonon, bond_cutoff=5)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.phonon.png", dpi=300)

    fig, ax, anim = animate_phonon_xy(phonon, bond_cutoff=5)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.phonon_animation.gif", dpi=300, writer="pillow"
    )

    fig, ax, anim = animate_phonon_xyz(phonon, bond_cutoff=5)
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    anim.save(
        "./examples/figures/2d_surface.phonon_3d_animation.side.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax = plt.subplots()
    result = get_dispersion_path(
        system,
        DispersionPath(
            points=(
                CubicPoint.GAMMA.value,
                (20, CubicPoint.X.value),
                (20, CubicPoint.M.value),
                (20, CubicPoint.Y.value),
                (20, CubicPoint.GAMMA.value),
            ),
        ),
    )
    fig, ax, line = plot_dispersion_path(result, branch=0, ax=ax)
    line.set_label("Branch 0")
    fig, ax, line = plot_dispersion_path(result, branch=1, ax=ax)
    line.set_label("Branch 1")
    fig, ax, line = plot_dispersion_path(result, branch=2, ax=ax)
    line.set_label("Branch 2")
    ax.legend()

    ax.set_title("Phonon Dispersion for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.dispersion.png", dpi=300)
