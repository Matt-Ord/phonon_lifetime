from phonon_lifetime import cell, system


def test_strain_as_supercell_shape() -> None:
    primitive = cell.build.cubic(mass=10, structure="simple", distance=1.0)
    strain = system.build.with_nearest_neighbor_forces(
        primitive,
        spring_constant=1.0,
        periodic=(True, False, False),
        cutoff=1.1,
    )

    first_supercell = system.as_supercell(strain, n_repeats=(5, 4, 1))
    second_supercell = system.as_supercell(first_supercell, n_repeats=(3, 3, 1))

    assert first_supercell.strain_repeats == (3, 1, 1)
    assert second_supercell.strain_repeats == (3, 1, 1)
