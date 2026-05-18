import matplotlib.pyplot as plt

from phonon_lifetime import system
from phonon_lifetime.cell import build
from phonon_lifetime.phonon import (
    CubicPoint,
    DispersionPath,
    animate_phonon_1d_x,
    as_gamma_phonon,
    as_supercell_phonon,
    get_dispersion_path,
    get_mesh_phonon,
    plot_dispersion_path,
    plot_phonon_1d_x,
)

if __name__ == "__main__":
    cell = build.cubic(mass=10, distance=1.0, structure="simple")
    strain = system.build.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, False, False), threshold=(0.0, 1.1)
    )

    phonon = get_mesh_phonon(strain, n_repeats=(51, 1, 1), q=(1, 0, 0), branch=2)
    phonon = as_gamma_phonon(phonon)
    phonon = as_supercell_phonon(phonon, n_repeats=(3, 1, 1))
    fig, ax, line = plot_phonon_1d_x(phonon)
    line.set_label("51x1x1 Phonon")

    phonon = get_mesh_phonon(strain, n_repeats=(51 * 3, 1, 1), q=(3, 0, 0), branch=2)
    phonon = as_gamma_phonon(phonon)
    fig, ax, line = plot_phonon_1d_x(phonon, ax=ax)
    line.set_linestyle("--")
    line.set_label("153x1x1 Phonon")

    ax.legend()
    ax.set_title(
        "Phonon Mode for 1D Chain\n. The mode calculated on the supercell (dashed line) "
        "should match\nthe mode calculated on the primitive cell (solid line)."
    )
    fig.savefig("./examples/figures/1d_chain.phonon.png", dpi=300)

    fig, ax, anim = animate_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain")
    anim.save(
        "./examples/figures/1d_chain.phonon_animation.gif", dpi=300, writer="pillow"
    )

    result = get_dispersion_path(
        strain,
        DispersionPath(
            points=(
                CubicPoint.MINUS_X.value,
                (20, CubicPoint.GAMMA.value),
                (20, CubicPoint.X.value),
            ),
        ),
    )
    fig, ax = plt.subplots()
    fig, ax, line = plot_dispersion_path(result, branch=0, ax=ax)
    line.set_label("Branch 0")
    fig, ax, line = plot_dispersion_path(result, branch=1, ax=ax)
    line.set_label("Branch 1")
    fig, ax, line = plot_dispersion_path(result, branch=2, ax=ax)
    line.set_label("Branch 2")
    ax.legend()
    ax.set_title("Phonon Dispersion Relation for 1D Chain")
    fig.savefig("./examples/figures/1d_chain.dispersion.png", dpi=300)
