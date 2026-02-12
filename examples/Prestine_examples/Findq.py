import numpy as np

from phonon_lifetime.Normal_Mode_2 import System, calculate_normal_modes

sys2d = System(
    element="Ni",
    cell=np.diag([1.0, 1.0, 1.0]),
    n_repeats=(15, 15, 1),
    spring_constant=(1.0, 1.0, 1.0),
)

res = calculate_normal_modes(sys2d)
res1 = res.get_modes_at_branch(branch=2)
q = res1.q_vals  # (Nq, 3) q points
evec = res1.modes  # (natom*3,3)
print(evec.shape)
tol = 1e-3

qx_indices = np.where(
    # np.isclose(q[:, 0], 0.0, atol=tol) & ~np.isclose(evec[:, 0], 0.0, atol=tol)
    np.isclose(q[:, 0], 0, atol=tol)
)[0]
print(qx_indices)
print(q[qx_indices])
