from phonon_lifetime import pristine, system
from phonon_lifetime.modes import animate_mode_xy

if __name__ == "__main__":
    graphene = system.build.graphene(mass=10, n_repeats=(6, 6, 1))
    graphene = pristine.with_ase_forces(graphene, periodic=(True, True, False))

    fig, ax, _ = system.plot_xyz(graphene, bond_cutoff=6)
    ax.set_title("Graphene Lattice")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    ax.set_aspect("equalxy")  # cspell: disable-line
    fig.savefig("./examples/figures/graphene.lattice.side.png", dpi=300)

    fig, ax, _ = system.plot_xy(graphene, bond_cutoff=6)
    ax.set_title("Graphene Lattice (2D Projection)")
    fig.savefig("./examples/figures/graphene.lattice.above.png", dpi=300)

    result = graphene.get_modes()
    mode = result.select_mode(branch=4, q=(1, 0, 0))

    fig, ax, anim = animate_mode_xy(mode, bond_cutoff=6, scale_displacement=0.2)
    anim.save(
        "./examples/figures/graphene.mode_3d_animation.above.gif",
        dpi=300,
        writer="pillow",
    )
    ax.set_aspect("equal")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    anim.save(
        "./examples/figures/graphene.mode_3d_animation.side.gif",
        dpi=300,
        writer="pillow",
    )
