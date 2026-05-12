# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "phonon_lifetime",
# ]
# [tool.uv.sources]
# phonon_lifetime = { git = "https://github.com/Matt-Ord/phonon_lifetime.git", rev = "58234049020dd9dcdc5b001479ccec6d8b71e08c" }
# ///

from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.phonon import (
    animate_phonon_xyz,
    as_gamma_phonon,
    get_mesh_phonons,
)
from phonon_lifetime.system import build as build_system

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        cell, spring_constant=1.0, periodic=(True, True, False), cutoff=1.1
    )
    result = get_mesh_phonons(system, n_repeats=(11, 11, 1))

    phonon = result.select_phonon(branch=2, iq=(1, 0, 0))
    phonon = as_gamma_phonon(phonon)

fig, ax, anim = animate_phonon_xyz(phonon, bond_cutoff=5)
ax.set_title("Phonon Mode for 2D Surface")
anim.save("2d_surface.phonon_animation.gif", dpi=300, writer="pillow")
fig, ax, anim = animate_phonon_xyz(phonon, bond_cutoff=5)

fig.set_size_inches(10, 8)
ax.set_position([0.05, 0.05, 0.9, 0.85])
ax.dist = 3
ax.set_box_aspect([1, 1, 0.7])
ax.view_init(elev=20, azim=90)
ax.set_zticks([])
ax.set_title("Phonon Mode in 2D")

anim.save(
    "2d_surface.phonon_3d_animation.side.gif",
    dpi=300,
    writer="pillow",
)
