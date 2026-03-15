import ase.build
import numpy as np

from phonon_lifetime.cell import SuperCell, from_ase_atoms
from phonon_lifetime.cell import build as build_cell


def test_atom_fractions_square() -> None:

    cell = build_cell.cubic(mass=10, distance=1, structure="simple")
    cell = SuperCell(cell, n_repeats=(7, 3, 5))
    fractions = cell.primitive_cell.atom_fractions
    print(fractions)
    np.testing.assert_array_almost_equal(fractions, [[0.0, 0.0, 0.0]])

    all_fractions = cell.atom_fractions
    expected_fractions = (
        np.asarray(
            np.meshgrid(
                np.arange(cell.n_repeats[0]) / cell.n_repeats[0],
                np.arange(cell.n_repeats[1]) / cell.n_repeats[1],
                np.arange(cell.n_repeats[2]) / cell.n_repeats[2],
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )
    np.testing.assert_array_almost_equal(all_fractions, expected_fractions)


def test_supercell_atom_fractions_square() -> None:

    cell = build_cell.cubic(mass=10, distance=1, structure="simple")
    cell = SuperCell(cell, n_repeats=(7, 3, 5))
    all_fractions = cell.atom_fractions
    expected_fractions = (
        np.asarray(
            np.meshgrid(
                np.arange(cell.n_repeats[0]) / cell.n_repeats[0],
                np.arange(cell.n_repeats[1]) / cell.n_repeats[1],
                np.arange(cell.n_repeats[2]) / cell.n_repeats[2],
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
    cell_raw = atoms.get_cell().array
    cell_raw[2, 2] = 1
    atoms.set_cell(cell_raw)
    repeat_atoms = atoms.repeat(n_repeats)
    cell = from_ase_atoms(atoms)
    cell = SuperCell(cell, n_repeats=n_repeats)

    actual = cell.primitive_cell.atom_fractions
    expected = atoms.get_scaled_positions()
    np.testing.assert_array_almost_equal(actual, expected)

    actual = cell.atom_fractions
    expected = repeat_atoms.get_scaled_positions()
    np.testing.assert_array_almost_equal(actual, expected)


def test_supercell_masses() -> None:
    n_repeats = (3, 1, 1)
    atoms = ase.build.graphene(a=2.46, thickness=3.35)
    atoms_cell = atoms.get_cell().array
    atoms_cell[2, 2] = 1
    atoms.set_cell(atoms_cell)
    atoms.set_masses((3, 5))
    repeat_atoms = atoms.repeat(n_repeats)
    cell = from_ase_atoms(atoms)
    cell = SuperCell(cell, n_repeats=n_repeats)

    actual = cell.masses
    expected = repeat_atoms.get_masses()
    np.testing.assert_array_almost_equal(actual, expected)

    repeat = SuperCell(cell, n_repeats=(2, 1, 1))
    actual = repeat.masses
    expected = repeat_atoms.repeat((2, 1, 1)).get_masses()
    np.testing.assert_array_almost_equal(actual, expected)


def test_supercell_symbols() -> None:
    n_repeats = (3, 1, 1)
    atoms = ase.build.graphene(a=2.46, thickness=3.35)
    atoms_cell = atoms.get_cell().array
    atoms_cell[2, 2] = 1
    atoms.set_cell(atoms_cell)
    atoms.set_chemical_symbols(("H", "C"))
    repeat_atoms = atoms.repeat(n_repeats)
    cell = from_ase_atoms(atoms)
    cell = SuperCell(cell, n_repeats=n_repeats)

    actual = cell.symbols
    expected = repeat_atoms.get_chemical_symbols()
    np.testing.assert_array_equal(actual, expected)

    repeat = SuperCell(cell, n_repeats=(2, 1, 1))
    actual = repeat.symbols
    expected = repeat_atoms.repeat((2, 1, 1)).get_chemical_symbols()
    np.testing.assert_array_equal(actual, expected)
