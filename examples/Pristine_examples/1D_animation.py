# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "phonon_lifetime",
#     "pillow>=11.0.0",
# ]
# [tool.uv.sources]
# phonon_lifetime = { git = "https://github.com/Matt-Ord/phonon_lifetime.git", rev = "58234049020dd9dcdc5b001479ccec6d8b71e08c" }
# ///

from phonon_lifetime.cell import build
from phonon_lifetime.phonon import (
    animate_phonon_1d_x,
    as_gamma_phonon,
    get_mesh_phonon,
)

from phonon_lifetime import system

if __name__ == "__main__":
    cell = build.cubic(mass=10, distance=1.0, structure="simple")
    strain = system.build.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )

    phonon = get_mesh_phonon(strain, n_repeats=(25 * 3, 1, 1), q=(1, 0, 0), branch=2)
    phonon = as_gamma_phonon(phonon)

    fig, ax, anim = animate_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain")
    anim.save("./1d_chain.phonon_animation.q1.gif", dpi=300, writer="pillow")

    phonon = get_mesh_phonon(strain, n_repeats=(25 * 3, 1, 1), q=(5, 0, 0), branch=2)
    phonon = as_gamma_phonon(phonon)

    fig, ax, anim = animate_phonon_1d_x(phonon)
    ax.set_title("Phonon Mode for 1D Chain")
    anim.save("./1d_chain.phonon_animation.q5.gif", dpi=300, writer="pillow")
