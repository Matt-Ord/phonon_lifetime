from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt

from phonon_lifetime.Normal_Mode_2 import (
    NormalModeResult,
    System,
    calculate_normal_modes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

pi = math.pi


# Default params (separate from function for flexible reuse)
DEFAULT_CRYSTAL_PARAMS: dict[str, Any] = {
    "Nx": 15,
    "Ny": 15,
    "element": "Ni",
    "cell": None,  # if None -> np.diag([1,1,1])
    "n_repeats": None,  # if None -> (Nx, Ny, 1)
    "spring_constant": (1.0, 1.0, 0.0),
}

DEFAULT_PLOT_PARAMS: dict[str, Any] = {
    "tol": 1e-10,
    "save_path": Path("./examples"),  # folder or file path are both allowed
    "dpi": 300,
    "show": False,
}


def display_2d_eigenvector(result: NormalModeResult) -> tuple[Figure, Axes]: ...


def plot_normal_mode(mode: NormalMode) -> tuple[Figure, Axes]:
    # Merge params
    cp = {**DEFAULT_CRYSTAL_PARAMS, **(crystal_params or {})}
    pp = {**DEFAULT_PLOT_PARAMS, **(plot_params or {})}

    Nx: int = int(cp["Nx"])
    Ny: int = int(cp["Ny"])
    element: str = str(cp["element"])
    cell = cp["cell"]
    n_repeats = cp["n_repeats"]
    spring_constant = cp["spring_constant"]

    float(pp["tol"])
    save_path = pp["save_path"]
    dpi: int = int(pp["dpi"])
    bool(pp["show"])

    # Define 2D lattice
    if cell is None:
        cell = np.diag([1.0, 1.0, 1.0])
    if n_repeats is None:
        n_repeats = (Nx, Ny, 1)

    lattice = System(
        element=element,
        cell=cell,
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    result = calculate_normal_modes(lattice)
    b1 = result.modes
    print(b1.shape)
    # Select one branch
    b2 = result.get_modes_at_branch(branch=branch)

    # Calculate Eigenfrequecies
    Omega = b2.omega
    eigen_vectors = b2.modes  # (Nq, 3*Nx*Ny)
    q_vals = b2.q_vals  # (Nq, 3)
    # Pick the eigenvector at one q
    q = np.asarray(list(q), dtype=float).reshape(-1)
    if q.size != 3:
        msg = f"q must be length-3, got shape {q.shape}"
        raise ValueError(msg)

    # Find nearest q-point in the computed mesh
    dq = q_vals - q[None, :]
    # check if q is correct
    dq -= np.rint(dq)
    iq = int(np.argmin(np.sum(dq * dq, axis=1)))

    q_mesh = q_vals[iq]
    print("target q =", q, "picked q_mesh =", q_mesh, "delta(min-image) =", dq[iq])

    Omega_q = Omega[iq]
    print(Omega_q)
    q_mesh = q_vals[iq, :]  # (qx, qy, qz=0)

    eigen_vector = eigen_vectors[iq, :]  # (3*Nx*Ny,)

    Nx, Ny = lattice.n_repeats[0], lattice.n_repeats[1]
    a1 = np.asarray(lattice.cell[0], float)
    a2 = np.asarray(lattice.cell[1], float)

    # Reshape eigenvector to per-site displacement
    evec = np.asarray(eigen_vector)  # shape: (3,)

    u_real = np.zeros((Nx, Ny, 3), float)

    for ix in range(Nx):
        for iy in range(Ny):
            R = ix * a1 + iy * a2
            phase = np.cos(2 * pi * np.dot(q_mesh, R))  # Re[exp(i q·R)] at t=0
            ix * Ny + iy
            u_real[ix, iy, :] = np.real(evec * phase)

    X = np.zeros((Nx, Ny))
    Y = np.zeros((Nx, Ny))

    for ix in range(Nx):
        for iy in range(Ny):
            R = ix * a1 + iy * a2
            X[ix, iy] = R[0]
            Y[ix, iy] = R[1]

    u_real = mode.get_mode_vector()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.quiver(
        X, Y, u_real[:, :, 0], u_real[:, :, 1], angles="xy", scale_units="xy", scale=1.0
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Real-space Bloch displacement (t = 0), element={element}, branch={branch}, iq={iq}"
    )

    plt.tight_layout()

    # Save figure
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            out_path = save_path
        else:
            out_path = (
                save_path / f"2d_mode_displacement_{element}_branch{branch}_iq{iq}.png"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    else:
        out_path = None

    return {
        "fig": fig,
        "ax": ax,
        "iq": iq,
        "q_mesh": q_mesh,
        "omega": float(Omega[iq]),
        "eigen_vector": eigen_vector,
        "u_real": u_real,
        "X": X,
        "Y": Y,
        "lattice": lattice,
        "save_path": out_path,
    }


def plot_2d_eigenvector(
    *,
    branch: int,
    q: Iterable[float],
    crystal_params: dict[str, Any] | None = None,
    plot_params: dict[str, Any] | None = None,
):
    # Merge params
    cp = {**DEFAULT_CRYSTAL_PARAMS, **(crystal_params or {})}
    pp = {**DEFAULT_PLOT_PARAMS, **(plot_params or {})}

    Nx: int = int(cp["Nx"])
    Ny: int = int(cp["Ny"])
    element: str = str(cp["element"])
    cell = cp["cell"]
    n_repeats = cp["n_repeats"]
    spring_constant = cp["spring_constant"]

    float(pp["tol"])
    save_path = pp["save_path"]
    dpi: int = int(pp["dpi"])
    bool(pp["show"])

    # Define 2D lattice
    if cell is None:
        cell = np.diag([1.0, 1.0, 1.0])
    if n_repeats is None:
        n_repeats = (Nx, Ny, 1)

    lattice = System(
        element=element,
        cell=cell,
        n_repeats=n_repeats,
        spring_constant=spring_constant,
    )

    result = calculate_normal_modes(lattice)
    b1 = result.modes
    print(b1.shape)
    # Select one branch
    b2 = result.get_modes_at_branch(branch=branch)

    # Calculate Eigenfrequecies
    Omega = b2.omega
    eigen_vectors = b2.modes  # (Nq, 3*Nx*Ny)
    q_vals = b2.q_vals  # (Nq, 3)
    # Pick the eigenvector at one q
    q = np.asarray(list(q), dtype=float).reshape(-1)
    if q.size != 3:
        msg = f"q must be length-3, got shape {q.shape}"
        raise ValueError(msg)

    # Find nearest q-point in the computed mesh
    dq = q_vals - q[None, :]
    # check if q is correct
    dq -= np.rint(dq)
    iq = int(np.argmin(np.sum(dq * dq, axis=1)))

    q_mesh = q_vals[iq]
    print("target q =", q, "picked q_mesh =", q_mesh, "delta(min-image) =", dq[iq])

    Omega_q = Omega[iq]
    print(Omega_q)
    q_mesh = q_vals[iq, :]  # (qx, qy, qz=0)

    eigen_vector = eigen_vectors[iq, :]  # (3*Nx*Ny,)

    Nx, Ny = lattice.n_repeats[0], lattice.n_repeats[1]
    a1 = np.asarray(lattice.cell[0], float)
    a2 = np.asarray(lattice.cell[1], float)

    # Reshape eigenvector to per-site displacement
    evec = np.asarray(eigen_vector)  # shape: (3,)

    u_real = np.zeros((Nx, Ny, 3), float)

    for ix in range(Nx):
        for iy in range(Ny):
            R = ix * a1 + iy * a2
            phase = np.cos(2 * pi * np.dot(q_mesh, R))  # Re[exp(i q·R)] at t=0
            ix * Ny + iy
            u_real[ix, iy, :] = np.real(evec * phase)

    X = np.zeros((Nx, Ny))
    Y = np.zeros((Nx, Ny))

    for ix in range(Nx):
        for iy in range(Ny):
            R = ix * a1 + iy * a2
            X[ix, iy] = R[0]
            Y[ix, iy] = R[1]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.quiver(
        X, Y, u_real[:, :, 0], u_real[:, :, 1], angles="xy", scale_units="xy", scale=1.0
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Real-space Bloch displacement (t = 0), element={element}, branch={branch}, iq={iq}"
    )

    plt.tight_layout()

    # Save figure
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            out_path = save_path
        else:
            out_path = (
                save_path / f"2d_mode_displacement_{element}_branch{branch}_iq{iq}.png"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    else:
        out_path = None

    return {
        "fig": fig,
        "ax": ax,
        "iq": iq,
        "q_mesh": q_mesh,
        "omega": float(Omega[iq]),
        "eigen_vector": eigen_vector,
        "u_real": u_real,
        "X": X,
        "Y": Y,
        "lattice": lattice,
        "save_path": out_path,
    }


if __name__ == "__main__":
    out = plot_2d_eigenvector(
        branch=0,
        q=(1 / 3, 2 / 3, 0.0),
        crystal_params={
            "Nx": 3,
            "Ny": 3,
            "element": "Ni",
            "spring_constant": (1.0, 1.0, 0.0),
        },
        plot_params={
            "save_path": Path("./examples/2D_Results/Eigenvectors/"),
            "show": False,
        },
    )
    print("saved to", out["save_path"])
