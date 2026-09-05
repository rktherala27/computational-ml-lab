from dataclasses import dataclass

import torch

from pinn_model import DuffingPINN

@dataclass
class DuffingParameters:
    c_true  : float
    k       : float
    alpha   : float


def duffing_residuals(model,t,params):

    """
    Compute residuals for the first-order Duffing oscillator system.

    The PINN predicts:
        x_hat(t): displacement
        v_hat(t): velocity

    Physical equations:
        dx/dt = v

        dv/dt = -c_true*v - k*x - alpha*x^3

    A perfect solution gives residuals equal to zero.
    """

    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    prediction = model(t)

    # Keep shape [N, 1], rather than reducing it to [N].
    x_hat = prediction[:, 0].view(-1,1)
    v_hat = prediction[:, 1].view(-1,1)

    # d(x_hat)/dt
    dx_dt = torch.autograd.grad(outputs=x_hat,
        inputs=t, grad_outputs=torch.ones_like(x_hat),
        create_graph=True, retain_graph=True)[0]

    # d(v_hat)/dt
    dv_dt = torch.autograd.grad(outputs=v_hat, inputs=t,
        grad_outputs=torch.ones_like(v_hat), create_graph=True, retain_graph=True)[0]

    # Residual of dx/dt - v = 0
    residual_x = dx_dt - v_hat

    # Residual of:
    # dv/dt + delta*v + alpha*x + beta*x^3 - gamma*cos(omega*t) = 0
    residual_v = (dv_dt + params.c_true * v_hat + params.k * x_hat
        + params.alpha * x_hat**3)

    return residual_x, residual_v

def physics_loss(model, t, params):
    """
    Mean-squared error of both Duffing equation residuals.
    """

    residual_x, residual_v = duffing_residuals(model, t, params)

    loss_x = torch.mean(residual_x**2)
    loss_v = torch.mean(residual_v**2)

    return loss_x + loss_v


if __name__ == "__main__":

    """
    This program demonstrates a single forward pass and calculates
    physics residuals and loss for an inverse PINN solving the
    unforced Duffing oscillator.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporary smoke-test values.
    params = DuffingParameters(c_true = 0.30, k = 1.00, alpha = 0.50)

    model = DuffingPINN().to(device)

    # Collocation points where the PINN will be required to obey physics.
    t_collocation = (torch.linspace(0.0, 20.0, 100).view(-1, 1).to(device).requires_grad_(True))

    residual_x, residual_v = duffing_residuals(model=model,t=t_collocation, params=params)

    loss = physics_loss(model=model, t=t_collocation, params=params)

    print(f"Device: {device}")
    print(f"Time shape: {t_collocation.shape}")
    print(f"x-equation residual shape: {residual_x.shape}")
    print(f"v-equation residual shape: {residual_v.shape}")
    print(f"Physics loss: {loss.item():.6f}")