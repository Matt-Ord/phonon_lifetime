from phonon_lifetime.modes import animate_mode_xy, repeat_mode
from phonon_lifetime.pristine import build_graphene_system
from phonon_lifetime.system import plot_system_xy, plot_system_xyz

if __name__ == "__main__":
    system = build_graphene_system(
        mass=10,
        n_repeats=(3, 3, 1),
        spring_constant=1,
    )

    fig, ax, _ = plot_system_xyz(system)
    ax.set_title("Graphene Lattice")
    ax.view_init(elev=20, azim=90)  # View from the side (20 degrees above the plane)
    ax.set_aspect("equalxy")  # cspell: disable-line
    fig.savefig("./examples/figures/graphene.lattice.side.png", dpi=300)

    fig, ax, _ = plot_system_xy(system)
    ax.set_title("Graphene Lattice (2D Projection)")
    fig.savefig("./examples/figures/graphene.lattice.above.png", dpi=300)

    result = system.get_modes()
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
