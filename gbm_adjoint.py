"""Scalar GBM replication of Li et al. (2020), Algorithm 3 and Figure 7(a).

Run with `python gbm_adjoint.py` after installing torch and matplotlib.
Brownian increments are stored as a tensor of shape (steps, paths).
"""

import math
from pathlib import Path

import torch


def exact_gbm(increments, dt, s0, mu, sigma):
    """Return the terminal state and its pathwise S0 and sigma derivatives."""
    time = increments.shape[0] * dt
    brownian = increments.sum(dim=0)
    growth = torch.exp((mu - 0.5 * sigma**2) * time + sigma * brownian)
    terminal = s0 * growth
    return terminal, growth, terminal * (brownian - sigma * time)


def milstein_gbm(increments, dt, s0, mu, sigma):
    """Solve dS = mu*S dt + sigma*S dW on the supplied grid."""
    state = s0
    for dw in increments:
        correction = 0.5 * sigma**2 * (dw.square() - dt)
        state = state * (1 + mu * dt + sigma * dw + correction)
    return state


def backprop_gradients(increments, dt, s0, mu, sigma):
    # Independent copies give one derivative per path in a single autograd call.
    spot = torch.full_like(increments[0], s0, requires_grad=True)
    volatility = torch.full_like(increments[0], sigma, requires_grad=True)
    terminal = milstein_gbm(increments, dt, spot, mu, volatility)
    return torch.autograd.grad(terminal.sum(), (spot, volatility))


@torch.no_grad()
def adjoint_gradients(increments, dt, terminal, terminal_grad, mu, sigma):
    """Return dLoss/dS0 and dLoss/dsigma by solving the backward SDE."""
    state = terminal
    state_adjoint = terminal_grad
    sigma_adjoint = torch.zeros_like(terminal)
    drift = mu - 0.5 * sigma**2
    for step in range(increments.shape[0] - 1, -1, -1):
        # Use the original increment with the signs already in the reverse
        # Stratonovich coefficients G = (-sigma*S, sigma*A, S*A).
        # The Milstein correction uses DG @ G = (sigma**2*S, sigma**2*A, 0).
        dw = increments[step]
        linear = drift * dt + sigma * dw
        correction = 0.5 * (sigma * dw).square()
        sigma_adjoint = sigma_adjoint + state * state_adjoint * (dw - sigma * dt)
        state_adjoint = state_adjoint * (1 + linear + correction)
        state = state * (1 - linear + correction)
    return state_adjoint, sigma_adjoint


def main():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s0, mu, sigma, maturity = 1.0, 0.05, 0.20, 1.0
    step_counts = (8, 16, 32, 64, 128, 256, 512)
    paths = 2048
    finest = step_counts[-1]
    rng = torch.Generator().manual_seed(2020)
    fine_noise = torch.randn(finest, paths, generator=rng, dtype=torch.float64)
    fine_noise *= math.sqrt(maturity / finest)

    errors = {"Adjoint": [], "Backprop": []}
    step_sizes = []
    print("Steps   Adjoint S0   Backprop S0   Adjoint sigma   Backprop sigma")
    for steps in step_counts:
        dt = maturity / steps
        step_sizes.append(dt)
        increments = fine_noise.reshape(steps, finest // steps, paths).sum(dim=1)
        _, exact_s0, exact_sigma = exact_gbm(increments, dt, s0, mu, sigma)
        terminal = milstein_gbm(increments, dt, s0, mu, sigma)
        gradients = {
            "Adjoint": adjoint_gradients(
                increments, dt, terminal, torch.ones_like(terminal), mu, sigma
            ),
            "Backprop": backprop_gradients(increments, dt, s0, mu, sigma),
        }
        for method, (grad_s0, grad_sigma) in gradients.items():
            rmse_s0 = (grad_s0 - exact_s0).square().mean().sqrt().item()
            rmse_sigma = (grad_sigma - exact_sigma).square().mean().sqrt().item()
            errors[method].append((rmse_s0, rmse_sigma))
        adjoint_s0, adjoint_sigma = errors["Adjoint"][-1]
        backprop_s0, backprop_sigma = errors["Backprop"][-1]
        print(f"{steps:5d}   {adjoint_s0:10.3e}   {backprop_s0:11.3e}   "
              f"{adjoint_sigma:13.3e}   {backprop_sigma:14.3e}")

    figure = Figure(figsize=(8.5, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle("GBM gradient convergence under Milstein discretization")
    axes = figure.subplots(1, 2)
    for index, (axis, parameter) in enumerate(zip(axes, ("$S_0$", r"$\sigma$"))):
        for method, style in (("Adjoint", "o-"), ("Backprop", "s--")):
            axis.loglog(
                step_sizes, [row[index] for row in errors[method]],
                style, label=method, markerfacecolor="none",
            )
        axis.set(xlabel=r"Step size $\Delta t$", ylabel=f"Gradient RMSE for {parameter}")
        axis.grid(alpha=0.25, which="major")
        axis.legend()
    axes[0].set_title("Curves overlap for $S_0$")
    output = Path(__file__).with_name("gbm_convergence.png")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
