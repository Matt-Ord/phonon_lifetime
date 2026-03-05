from phonon_lifetime import pristine, system
from phonon_lifetime.modes import animate_mode_xy, repeat_mode

if __name__ == "__main__":
    graphene = system.build.graphene(mass=10, n_repeats=(3, 3, 1))
    graphene = pristine.with_nearest_neighbor_force(
        graphene, spring_constant=1, periodic=(True, True, False)
    )

    fig, ax, _ = system.plot_xyz(graphene)
    ax.set_title("Graphene Lattice")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    ax.set_aspect("equalxy")  # cspell: disable-line
    fig.savefig("./examples/figures/graphene.lattice.side.png", dpi=300)

    fig, ax, _ = system.plot_xy(graphene)
    ax.set_title("Graphene Lattice (2D Projection)")
    fig.savefig("./examples/figures/graphene.lattice.above.png", dpi=300)

    result = graphene.get_modes()
    mode = result.select_mode(branch=4, q=(1, 0, 0))

    fig, ax, anim = animate_mode_xy(repeat_mode(mode, (2, 3, 1)))
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
