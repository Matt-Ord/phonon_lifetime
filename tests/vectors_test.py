import numpy as np

from phonon_lifetime.cell import SuperCell
from phonon_lifetime.cell import build as build_cell
from phonon_lifetime.defect import MassDefect, with_mass_defect
from phonon_lifetime.phonon import get_gamma_phonons
from phonon_lifetime.system import with_zero_forces


def test_mass_defect_vectors() -> None:
    cell = build_cell.cubic(mass=10, distance=1, structure="simple")
    cell = SuperCell(cell, n_repeats=(7, 1, 1))

    system = with_zero_forces(cell)
    defect = with_mass_defect(pristine=system, defects=MassDefect(defects=[]))

    modes = get_gamma_phonons(defect)
    vectors = modes.vectors

    for i in range(modes.n_modes):
        np.testing.assert_array_equal(vectors[i].reshape(-1, 3), modes[i].vector)
