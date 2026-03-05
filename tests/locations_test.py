import ase.build
import numpy as np

from phonon_lifetime.pristine import from_ase_atoms
from phonon_lifetime.system import (
    build,
    get_atom_fractions,
    get_atom_supercell_fractions,
)


def test_atom_fractions_square() -> None:

    system = build.cubic(mass=10, distance=1, n_repeats=(7, 3, 5), structure="simple")
    fractions = system.primitive_atom_fractions
    np.testing.assert_array_almost_equal(fractions, [[0.0, 0.0, 0.0]])

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
    np.testing.assert_array_almost_equal(all_fractions, expected_fractions)


def test_supercell_atom_fractions_square() -> None:

    system = build.cubic(mass=10, distance=1, n_repeats=(7, 3, 5), structure="simple")
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
    np.testing.assert_array_almost_equal(all_fractions, expected_fractions)


def test_supercell_atom_fractions_graphene() -> None:

    n_repeats = (3, 1, 1)
    atoms = ase.build.graphene(a=2.46, thickness=3.35)
    repeat_atoms = atoms.repeat(n_repeats)
    system = from_ase_atoms(atoms, n_repeats=n_repeats)

    actual = system.primitive_atom_fractions
    expected = atoms.get_scaled_positions()
    np.testing.assert_array_almost_equal(actual, expected)

    actual = get_atom_supercell_fractions(system)
    expected = repeat_atoms.get_scaled_positions()
    np.testing.assert_array_almost_equal(actual, expected)
