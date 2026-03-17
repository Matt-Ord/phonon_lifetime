import matplotlib.pyplot as plt

from phonon_lifetime import cell, system
from phonon_lifetime.phonon import (
    DispersionPath,
    HexagonalPoint,
    animate_phonon_xy,
    as_gamma_phonon,
    as_supercell_phonon,
    get_dispersion_path,
    get_mesh_phonon,
    plot_dispersion_path,
)
from phonon_lifetime.system import as_supercell

if __name__ == "__main__":
    graphene = cell.build.graphene(mass=10)
    strian = system.build.with_ase_forces(
        graphene, periodic=(True, True, False), n_repeats=(4, 4, 1)
    )

    strian_supercell = as_supercell(strian, n_repeats=(5, 5, 1))
    fig, ax, _ = system.plot_xyz(strian_supercell, bond_cutoff=6)
    ax.set_title("Graphene Lattice")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    ax.set_aspect("equalxy")  # cspell: disable-line
    fig.savefig("./examples/figures/graphene.lattice.side.png", dpi=300)

    fig, ax, _ = system.plot_xy(strian_supercell, bond_cutoff=6)
    ax.set_title("Graphene Lattice (2D Projection)")
    fig.savefig("./examples/figures/graphene.lattice.above.png", dpi=300)

    phonon = get_mesh_phonon(strian, n_repeats=(3, 3, 1), q=(1, 0, 0), branch=4)
    phonon = as_gamma_phonon(phonon)
    phonon = as_supercell_phonon(phonon, n_repeats=(2, 2, 1))

    fig, ax, anim = animate_phonon_xy(phonon, bond_cutoff=6, scale_displacement=0.2)
    anim.save(
        "./examples/figures/graphene.mode_3d_animation.above.gif",
        dpi=300,
        writer="pillow",
    )
    ax.set_aspect("equal")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    anim.save(
        "./examples/figures/graphene.mode_3d_animation.side.gif",
        dpi=300,
        writer="pillow",
    )

    fig, ax = plt.subplots()

    result = get_dispersion_path(
        strian,
        DispersionPath(
            points=(
                HexagonalPoint.GAMMA.value,
                (20, HexagonalPoint.M.value),
                (20, HexagonalPoint.K.value),
                (20, HexagonalPoint.GAMMA.value),
            ),
        ),
    )
    for i in range(6):
        fig, ax, line = plot_dispersion_path(result, branch=i, ax=ax)
        line.set_label(f"Branch {i}")

    ax.legend()

    ax.set_title("Phonon Dispersion for Graphene")
    fig.savefig("./examples/figures/graphene.dispersion.png", dpi=300)
