import numpy as np

from examples.Prestine_examples.Plot_Eigenfrequencies import (
    plot_branch_frequency_scatter,
)
from phonon_lifetime.Defected_Normal_Mode import (
    System,
    _buil
    calculate_normal_modes,
)

# fc = _build_force_constant_matrix(
#     System(
#         cell=np.diag([3.0, 3.0, 3.0]),
#         spring_constant=(1.0, 1.0, 0.0),
#         symbols=["Ni"] * 8,  # 8 symbols
#         scaled_positions=[
#             [0.0, 0.0, 0.0],
#             [1 / 3, 0.0, 0.0],
#             [2 / 3, 0.0, 0.0],
#             [0.0, 1 / 3, 0.0],
#             # Vacancy
#             [2 / 3, 1 / 3, 0.0],
#             [0.0, 2 / 3, 0.0],
#             [1 / 3, 2 / 3, 0.0],
#             [2 / 3, 2 / 3, 0.0],
#         ],
#     )
# )
# print(fc[:, :, 1, 1])
result = calculate_normal_modes(
    System(
        cell=np.diag([3.0, 3.0, 3.0]),
        spring_constant=(1.0, 1.0, 0.0),
        symbols=["Ni"] * 8,  # 8 symbols
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0],
            [2 / 3, 0.0, 0.0],
            [0.0, 1 / 3, 0.0],
            # Vacancy
            [2 / 3, 1 / 3, 0.0],
            [0.0, 2 / 3, 0.0],
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 2 / 3, 0.0],
        ],
    )
)
Modes = result.get_modes_at_branch(0)
Omega = Modes.omega
plot_branch_frequency_scatter(
    system=System(
        cell=np.diag([3.0, 3.0, 3.0]),
        spring_constant=(1.0, 1.0, 0.0),
        symbols=["Ni"] * 8,  # 8 symbols
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 3, 0.0, 0.0],
            [2 / 3, 0.0, 0.0],
            [0.0, 1 / 3, 0.0],
            # Vacancy
            [2 / 3, 1 / 3, 0.0],
            [0.0, 2 / 3, 0.0],
            [1 / 3, 2 / 3, 0.0],
            [2 / 3, 2 / 3, 0.0],
        ],
    ),
    branch=1,
    save_path="./Defected_examples/Eigenfrequency of branch {branch}.png",
)
