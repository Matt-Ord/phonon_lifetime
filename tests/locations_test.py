import numpy as np

from phonon_lifetime.pristine import PristineSystem
from phonon_lifetime.system import get_atom_fractions, get_atom_supercell_fractions


def test_atom_fractions_square() -> None:
    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(7, 3, 5),
        spring_constant=(0.0, 0.0, 0.0),
    )
    fractions = system.primitive_atom_fractions
    np.testing.assert_array_equal(fractions, [[0.0, 0.0, 0.0]])

    all_fractions = get_atom_fractions(system)
    expected_fractions = (
        np.asarray(
            np.meshgrid(
                np.arange(system.n_repeats[0]),
                np.arange(system.n_repeats[1]),
                np.arange(system.n_repeats[2]),
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )
    np.testing.assert_array_equal(all_fractions, expected_fractions)


def test_supercell_atom_fractions_square() -> None:
    system = PristineSystem.from_spring_constant(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(7, 3, 5),
        spring_constant=(0.0, 0.0, 0.0),
    )
    all_fractions = get_atom_supercell_fractions(system)
    expected_fractions = (
        np.asarray(
            np.meshgrid(
                np.arange(system.n_repeats[0]) / system.n_repeats[0],
                np.arange(system.n_repeats[1]) / system.n_repeats[1],
                np.arange(system.n_repeats[2]) / system.n_repeats[2],
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )
    np.testing.assert_array_equal(all_fractions, expected_fractions)
