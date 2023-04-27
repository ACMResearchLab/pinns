import torch
import torch.nn as nn
from pinn_functional import f, dfdx, d2fdx2

R = 1.0  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.5  # boundary condition value


def loss_fn(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

    # interior loss
    print("pre-worked?")
    f_value = f(x, params)
    print("pre-interior")
    interior = dfdx(x, params) - R * f_value * (1 - f_value)
    print("post-interor")

    # boundary loss
    x0 = X_BOUNDARY
    f0 = F_BOUNDARY
    x_boundary = torch.tensor([x0])
    f_boundary = torch.tensor([f0])
    boundary = f(x_boundary, params) - f_boundary

    loss = nn.MSELoss()
    loss_value = loss(interior, torch.zeros_like(interior)) + loss(
        boundary, torch.zeros_like(boundary)
    )

    return loss_value