import numpy as np

from phonon_lifetime.defect import (
    MassDefect,
    MassDefectSystem,
)
from phonon_lifetime.pristine import PristineSystem
from phonon_lifetime.wannier import plot_wannier_vector

if __name__ == "__main__":
    system = PristineSystem(
        mass=10,
        primitive_cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(101, 1, 1),
        spring_constant=(1, 0.0, 0.0),
    )

    modes = MassDefectSystem(
        pristine=system, defect=MassDefect(defects=[(9.8, 0)])
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
