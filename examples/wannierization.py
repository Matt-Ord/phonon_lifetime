from phonon_lifetime import pristine
from phonon_lifetime.defect import (
    MassDefect,
    MassDefectSystem,
)
from phonon_lifetime.system import build
from phonon_lifetime.wannier import plot_wannier_vector

if __name__ == "__main__":
    system = build.cubic(
        mass=10, distance=1.0, n_repeats=(101, 1, 1), structure="simple"
    )
    system = pristine.with_nearest_neighbor_forces(
        system, spring_constant=1.0, periodic=(True, False, False), cutoff=1.1
    )

    modes = MassDefectSystem(
        pristine=system, defect=MassDefect(defects=[(None, 9.8, 0)])
    ).get_modes()

    # We plot wannier modes - but they are rather uninteresting
    # if we include all the modes - they are simply the single atom
    # displacements!
    # TODO: if we truncate the modes, what do the  # noqa: FIX002
    # Wannier vectors look like? Do they look as expected?
    # The high-frequency modes are maybe not excited in the
    # real system - and we would therefore see a finite length scale in the Wannier vectors?
    fig, ax = plot_wannier_vector(modes, idx=0)
    fig.savefig("./examples/figures/1d_chain.wannier_vector.png", dpi=300)
