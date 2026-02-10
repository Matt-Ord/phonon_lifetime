from __future__ import annotations

import math

pi = math.pi
from pathlib import Path

import numpy as np

from phonon_lifetime.Normal_Mode_2 import (
    System,
    calculate_normal_modes,
)

if __name__ == "__main__":
    # Define 2D lattice
    Nx, Ny = 5, 5

    lattice = System(
        element="Ni",
        cell=np.diag([1.0, 1.0, 1.0]),
        n_repeats=(Nx, Ny, 1),
        spring_constant=(1.0, 1.0, 0.0),
    )

    result = calculate_normal_modes(lattice)
    b1 = result.modes
    # Pick one branch
    b2 = result.get_ModesAtBranch(branch=2)
    # Calculate Eigenfrequecies, where Nq=Nx*Ny
    Omega = b2.omega
    print("Omega shape", Omega.shape)
    eigen_vectors = b2.modes  # shape: (Nq, 3*Nx*Ny)
    print("eigvec shape", eigen_vectors.shape)
    q_vals = b2.q_vals  # shape: (Nq, 3)
    print("q_vales shape", q_vals.shape)
    # Pick the eigenvector at one q
    iq = 10
    q = q_vals[iq, :]  # (qx, qy, qz=0)
    eigen_vector = eigen_vectors[iq, :]  # (3*Nx*Ny,)
    # print(eigen_vector)
    print(q)


Nx, Ny = lattice.n_repeats[0], lattice.n_repeats[1]
a1 = np.asarray(lattice.cell[0], float)
a2 = np.asarray(lattice.cell[1], float)  # a2=(0,1,0)

from math import pi

import matplotlib.pyplot as plt

# Coordinats of A and B atoms
tau_A = np.array([0.0, 0.0, 0.0], dtype=float)
tau_B = 0.5 * a1  # real-space offset of B within the cell, a2 means y direction
taus = np.stack([tau_A, tau_B], axis=0)

# Reshape eigenvector into (natom, 3)
evec = np.asarray(eigen_vector)
evec = evec.reshape(2, 3)

# Allocate real-space displacement
u_real = np.zeros((Nx, Ny, 2, 3), float)

# Write the displacement
q_vec = np.asarray(q, float)

for ix in range(Nx):
    for iy in range(Ny):
        R_cell = ix * a1 + iy * a2
        theta = 2 * pi * np.dot(q_vec, R_cell)

        c = np.cos(theta)
        s = np.sin(theta)

        u_real[ix, iy, :, :] = np.real(evec) * c - np.imag(evec) * s
print(evec)
# Build plotting coordinates for both atoms
X = np.zeros((Nx, Ny, 2), float)
Y = np.zeros((Nx, Ny, 2), float)

for ix in range(Nx):
    for iy in range(Ny):
        R_cell = ix * a1 + iy * a2
        for b in range(2):
            R_atom = R_cell + taus[b]
            X[ix, iy, b] = R_atom[0]
            Y[ix, iy, b] = R_atom[1]

fig, ax = plt.subplots(figsize=(6, 6))

# Plot displacement
# Atom A
ax.quiver(
    X[:, :, 0],
    Y[:, :, 0],
    u_real[:, :, 0, 0],
    u_real[:, :, 0, 1],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    color="C0",
)

# Atom B
ax.quiver(
    X[:, :, 1],
    Y[:, :, 1],
    u_real[:, :, 1, 0],
    u_real[:, :, 1, 1],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    color="C1",
)

# Atom A positions
ax.scatter(
    X[:, :, 0].ravel(),
    Y[:, :, 0].ravel(),
    s=30,
    c="blue",
    marker="o",
    label="Atom A",
    zorder=3,
)

# Atom B positions
ax.scatter(
    X[:, :, 1].ravel(),
    Y[:, :, 1].ravel(),
    s=30,
    c="red",
    marker="o",
    label="Atom B",
    zorder=3,
)

# Save the figure
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Real-space Bloch displacement (t = 0)")
ax.legend(loc="upper right", frameon=True)

plt.tight_layout()

save_folder = Path("./examples")
save_folder.mkdir(parents=True, exist_ok=True)
save_path = save_folder / "2_atoms_percell.png"

plt.savefig(save_path, dpi=300)
plt.close()
