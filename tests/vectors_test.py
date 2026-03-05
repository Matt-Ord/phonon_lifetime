import numpy as np

from phonon_lifetime.defect import MassDefect, MassDefectSystem
from phonon_lifetime.system import build


def test_mass_defect_vectors() -> None:
    system = build.cubic(mass=10, distance=1, n_repeats=(7, 1, 1), structure="simple")

    defect = MassDefectSystem(pristine=system, defect=MassDefect(defects=[]))

    modes = defect.get_modes()
    vectors = modes.vectors

    for i in range(modes.n_modes):
        np.testing.assert_array_equal(vectors[i].reshape(-1, 3), modes[i].vector)
