from phonon_lifetime.cell import SuperCell
from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import (
    MassDefect,
    with_mass_defect,
)
from phonon_lifetime.phonon import get_gamma_phonons
from phonon_lifetime.system import build as build_system
from phonon_lifetime.wannier import plot_wannier_vector

if __name__ == "__main__":
    cell = build_cell.cubic(mass=10, distance=1.0, structure="simple")
    system = build_system.with_nearest_neighbor_forces(
        SuperCell(cell, n_repeats=(101, 1, 1)),
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    system = with_mass_defect(
        pristine=system, defects=MassDefect(defects=[(None, 9.8, 0)])
    )
    phonons = get_gamma_phonons(system)

    # We plot wannier modes - but they are rather uninteresting
    # if we include all the modes - they are simply the single atom
    # displacements!
    # TODO: if we truncate the modes, what do the  # noqa: FIX002
    # Wannier vectors look like? Do they look as expected?
    # The high-frequency modes are maybe not excited in the
    # real system - and we would therefore see a finite length scale in the Wannier vectors?
    fig, ax = plot_wannier_vector(phonons, idx=0)
    fig.savefig("./examples/figures/1d_chain.wannier_vector.png", dpi=300)
