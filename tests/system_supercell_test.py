import numpy as np

from phonon_lifetime import cell, system


def test_strain_as_supercell_shape() -> None:
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    strain = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, False, False),
        threshold=(0.0, 1.1),
    )

    first_supercell = system.as_supercell(strain, n_repeats=(5, 4, 1))
    second_supercell = system.as_supercell(first_supercell, n_repeats=(3, 3, 1))

    assert first_supercell.strain_repeats == (3, 1, 1)
    assert second_supercell.strain_repeats == (3, 1, 1)


def test_strain_as_supercell_twice() -> None:
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, False),
        threshold=(0.0, 1.1),
    )

    repeated_twice = system.as_supercell(
        system.as_supercell(base_system, n_repeats=(5, 3, 1)),
        n_repeats=(3, 1, 1),
    )
    repeated_once = system.as_supercell(base_system, n_repeats=(15, 3, 1))

    assert repeated_twice.cell.n_repeats == (3, 1, 1)
    assert repeated_once.cell.n_repeats == (15, 3, 1)
    assert repeated_twice.cell.n_atoms == repeated_once.cell.n_atoms
    assert repeated_twice.strain_repeats == repeated_once.strain_repeats
    np.testing.assert_array_equal(repeated_twice.strain, repeated_once.strain)


def test_strain_as_supercell_vs_supercell_forces() -> None:
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, False),
        threshold=(0.0, 1.1),
    )

    repeated_base_system = system.as_supercell(base_system, n_repeats=(5, 3, 1))

    repeated_system = system.with_nearest_neighbor_forces(
        repeated_base_system.cell,
        spring_constant=1.0,
        periodic=(True, True, False),
        threshold=(0.0, 1.1),
    )
    np.testing.assert_array_equal(
        repeated_base_system.strain_repeats, repeated_system.strain_repeats
    )
    np.testing.assert_array_equal(repeated_base_system.strain, repeated_system.strain)


def test_strain_as_supercell_vs_supercell_forces_graphene() -> None:
    primitive = cell.build.graphene(mass=10, distance=1.0)
    base_system = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, True, False),
        threshold=(0.0, 1.1),
    )

    repeated_base_system = system.as_supercell(base_system, n_repeats=(5, 3, 1))

    repeated_system = system.with_nearest_neighbor_forces(
        repeated_base_system.cell,
        spring_constant=1.0,
        periodic=(True, True, False),
        threshold=(0.0, 1.1),
    )
    np.testing.assert_array_almost_equal(
        repeated_base_system.strain_repeats, repeated_system.strain_repeats
    )
    np.testing.assert_array_almost_equal(
        repeated_base_system.strain, repeated_system.strain
    )
