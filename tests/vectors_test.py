import numpy as np

from phonon_lifetime.defect import MassDefect, MassDefectSystem
from phonon_lifetime.pristine import PristineSystem


def test_mass_defect_vectors() -> None:
    system = PristineSystem(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(7, 1, 1),
        spring_constant=(1, 0, 0),
    )

    defect = MassDefectSystem(pristine=system, defect=MassDefect(defects=[]))

    modes = defect.get_modes()
    vectors = modes.vectors

    for i in range(modes.n_modes):
        np.testing.assert_array_equal(
            vectors[i].reshape(-1, 3),
            modes[i].vector,
        )
