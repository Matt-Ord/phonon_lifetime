import matplotlib.pyplot as plt

from phonon_lifetime import pristine
from phonon_lifetime.modes import (
    animate_mode_xy,
    animate_mode_xyz,
    plot_mode_xy,
)
from phonon_lifetime.pristine import plot_dispersion_2d_xy
from phonon_lifetime.system import build

if __name__ == "__main__":
    system = build.cubic(
        mass=10, distance=1.0, n_repeats=(11, 11, 1), structure="simple"
    )
    system = pristine.with_nearest_neighbor_forces(
        system, spring_constant=1.0, periodic=(True, True, False), cutoff=1.1
    )
    result = system.get_modes()

    mode = result.select_mode(branch=2, q=(1, 0, 0))
    fig, ax, _ = plot_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.mode.png", dpi=300)

    fig, ax, anim = animate_mode_xy(mode, bond_cutoff=1.1)
    ax.set_title("Phonon Mode for 2D Surface")
    anim.save(
        "./examples/figures/2d_surface.mode_animation.gif", dpi=300, writer="pillow"
    )

    fig, ax, anim = animate_mode_xyz(mode, bond_cutoff=1.1)
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    anim.save(
        "./examples/figures/2d_surface.mode_3d_animation.side.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_dispersion_2d_xy(result.at_branch(2), ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.dispersion.2.png", dpi=300)

    fig, ax = plt.subplots()
    fig, ax, mesh = plot_dispersion_2d_xy(result.at_branch(1), ax=ax)
    fig.colorbar(mesh, label="Energy (THz)")

    ax.set_title("Phonon Dispersion Relation for 2D Surface")
    fig.savefig("./examples/figures/2d_surface.dispersion.1.png", dpi=300)
