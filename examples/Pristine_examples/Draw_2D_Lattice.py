import matplotlib.pyplot as plt
import numpy as np

# 11 x 11 square lattice
Nx, Ny = 11, 11
x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))

# defect at (2, 2): third atom from the lower-left corner
defect_pos = (2, 2)

fig, ax = plt.subplots(figsize=(7.5, 5.2))

# pristine atoms
ax.scatter(
    x,
    y,
    s=120,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.6,
    label="mass = 58.69",
    zorder=2,
)

# defect atom
ax.scatter(
    defect_pos[0],
    defect_pos[1],
    s=160,
    color="#C44E52",
    edgecolor="black",
    linewidth=0.8,
    label="mass = 10",
    zorder=3,
)

# guide grid / bonds
for i in range(Nx):
    ax.plot([i, i], [0, Ny - 1], color="0.85", lw=0.8, zorder=1)
for j in range(Ny):
    ax.plot([0, Nx - 1], [j, j], color="0.85", lw=0.8, zorder=1)

ax.set_aspect("equal")
ax.set_xlim(-0.7, Nx - 0.3)
ax.set_ylim(-0.7, Ny - 0.3)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Mass defected lattice")

ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
ax.tick_params(direction="in")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
output = "examples/Pristine_examples/Defect_Lattice.png"
fig.savefig(output)
print(output)
