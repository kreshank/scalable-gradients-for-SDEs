"""GBM check-in: gradient validation, call Greeks and memory scaling.

An adaptation of Li et al. (2020); option pricing is our extension.
Run `python gbm_adjoint.py --help` for experiment settings.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import torch


class StepNoise:
    """Fixed-grid CPU float64 noise with O(paths) memory and random step access.

    Each step uses its own local generator. Replaying a step does not depend on
    query order or change PyTorch's global RNG. This is not a Brownian tree:
    changing the grid does not preserve the underlying Brownian path.
    """

    def __init__(self, steps, paths, dt, seed=12345):
        self.shape = (steps, paths)
        self.dt = dt
        self.seed = seed

    def __getitem__(self, step):
        if not 0 <= step < self.shape[0]:
            raise IndexError(step)
        step_seed = (self.seed + step * 0x9E3779B97F4A7C15) % (2**63 - 1)
        generator = torch.Generator().manual_seed(step_seed)
        return torch.randn(self.shape[1], generator=generator, dtype=torch.float64) * math.sqrt(self.dt)

    def stored(self):
        # Fill directly rather than stack a list and temporarily double storage.
        increments = torch.empty(self.shape, dtype=torch.float64)
        for step in range(self.shape[0]):
            increments[step] = self[step]
        return increments


def validate_noise():
    """Check replay and complete gradients, including reverse queries and RNG isolation."""
    rng_before = torch.random.get_rng_state().clone()
    for seed in (0, 12345, 2020):
        for steps, paths in ((1, 2), (17, 127), (100, 256)):
            dt = 1 / steps
            noise = StepNoise(steps, paths, dt, seed)
            stored = noise.stored()
            for step in reversed(range(steps)):
                assert torch.equal(noise[step], stored[step]), (seed, step)
            terminal = milstein_gbm(stored, dt, 100.0, 0.05, 0.2)
            regenerated = milstein_gbm(noise, dt, 100.0, 0.05, 0.2)
            assert torch.equal(terminal, regenerated)
            for loss in ("square", "call"):
                weight = payoff_derivative(terminal, loss, 0.05, 1.0, 100.0)
                expected = adjoint_gradients(stored, dt, terminal, weight, 0.05, 0.2,
                                             return_state=True)
                actual = adjoint_gradients(noise, dt, regenerated, weight, 0.05, 0.2,
                                           return_state=True)
                for lhs, rhs in zip(expected, actual):
                    assert torch.equal(lhs, rhs), (seed, steps, loss)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    print("Noise replay passed: bitwise increments, terminal states, both gradients and reconstruction; global RNG unchanged.")


def black_scholes(s0, strike, rate, sigma, maturity):
    """Price, delta and vega (per unit volatility), for positive inputs."""
    if min(s0, strike, sigma, maturity) <= 0:
        raise ValueError("Spot, strike, volatility and maturity must be positive")
    root_time = math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (rate + sigma**2 / 2) * maturity) / (sigma * root_time)
    d2 = d1 - sigma * root_time
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    density = math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)
    return (s0 * cdf(d1) - strike * math.exp(-rate * maturity) * cdf(d2),
            cdf(d1), s0 * density * root_time)


def payoff(terminal, loss, rate, maturity, strike):
    if loss == "terminal":
        return terminal
    if loss == "square":
        return terminal.square()
    if loss == "call":
        return math.exp(-rate * maturity) * torch.where(
            terminal > strike, terminal - strike, torch.zeros_like(terminal)
        )
    raise ValueError(f"Unknown loss: {loss}")


def payoff_derivative(terminal, loss, rate, maturity, strike):
    if loss == "terminal":
        return torch.ones_like(terminal)
    if loss == "square":
        return 2 * terminal
    if loss == "call":
        return math.exp(-rate * maturity) * (terminal > strike).to(terminal.dtype)
    raise ValueError(f"Unknown loss: {loss}")


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


def backprop_gradients(increments, dt, s0, mu, sigma, loss="terminal", strike=100.0):
    # Independent copies give one derivative per path in a single autograd call.
    spot = torch.full_like(increments[0], s0, requires_grad=True)
    volatility = torch.full_like(increments[0], sigma, requires_grad=True)
    terminal = milstein_gbm(increments, dt, spot, mu, volatility)
    values = payoff(terminal, loss, mu, increments.shape[0] * dt, strike)
    return torch.autograd.grad(values.sum(), (spot, volatility))


@torch.no_grad()
def adjoint_gradients(increments, dt, terminal, terminal_grad, mu, sigma,
                      return_state=False):
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
    if return_state:
        return state_adjoint, sigma_adjoint, state
    return state_adjoint, sigma_adjoint


def estimate(values):
    return values.mean().item(), values.std(unbiased=True).item() / math.sqrt(values.numel())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth_validation(output, seeds, paths):
    """Couple nested grids so each seed has the same Brownian terminal value."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s0, rate, sigma, maturity = 100.0, 0.05, 0.2, 1.0
    step_counts = (10, 20, 50, 100, 200, 500, 1000)
    rows = []
    max_delta_difference = 0.0
    for seed in range(seeds):
        generator = torch.Generator().manual_seed(2020 + seed)
        fine = torch.randn(1000, paths, generator=generator, dtype=torch.float64)
        fine *= math.sqrt(maturity / 1000)
        for steps in step_counts:
            dt = maturity / steps
            increments = fine.reshape(steps, 1000 // steps, paths).sum(dim=1)
            exact, exact_s0, exact_sigma = exact_gbm(increments, dt, s0, rate, sigma)
            terminal = milstein_gbm(increments, dt, s0, rate, sigma)
            # Explicitly verify the algebraic Ito/Stratonovich equivalence.
            dw = increments[0]
            ito = 1 + rate * dt + sigma * dw + 0.5 * sigma**2 * (dw.square() - dt)
            strat = 1 + (rate - sigma**2 / 2) * dt + sigma * dw + 0.5 * sigma**2 * dw.square()
            torch.testing.assert_close(ito, strat, rtol=1e-14, atol=1e-14)
            adj_s0, adj_sigma, rebuilt = adjoint_gradients(
                increments, dt, terminal, 2 * terminal, rate, sigma, return_state=True
            )
            bp_s0, bp_sigma = backprop_gradients(
                increments, dt, s0, rate, sigma, loss="square"
            )
            torch.testing.assert_close(adj_s0, bp_s0, rtol=1e-11, atol=1e-10)
            max_delta_difference = max(max_delta_difference, (adj_s0 - bp_s0).abs().max().item())
            for method, gradients in (("Adjoint", (adj_s0, adj_sigma)),
                                      ("Backprop", (bp_s0, bp_sigma))):
                rows.append(dict(
                    seed=2020 + seed, steps=steps, dt=dt, method=method,
                    s0_rmse=(gradients[0] - 2 * exact * exact_s0).square().mean().sqrt().item(),
                    sigma_rmse=(gradients[1] - 2 * exact * exact_sigma).square().mean().sqrt().item(),
                    terminal_rmse=(terminal - exact).square().mean().sqrt().item(),
                    reconstruction_max=(rebuilt - s0).abs().max().item(),
                ))

    def average(method, steps, key):
        values = [row[key] for row in rows if row["method"] == method and row["steps"] == steps]
        return sum(values) / len(values)

    # Check aggregate improvement, rather than require monotonicity for every seed.
    for method in ("Adjoint", "Backprop"):
        for key in ("s0_rmse", "sigma_rmse", "terminal_rmse", "reconstruction_max"):
            assert average(method, 1000, key) < average(method, 10, key), (method, key)
    write_csv(output / "smooth_validation.csv", rows)
    figure = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, key, label in zip(figure.subplots(1, 2),
                                ("s0_rmse", "sigma_rmse"), ("S0", "sigma")):
        for method, marker in (("Adjoint", "o-"), ("Backprop", "s--")):
            axis.loglog([1 / steps for steps in step_counts],
                        [average(method, steps, key) for steps in step_counts],
                        marker, label=method, markerfacecolor="none")
        axis.set(xlabel="Step size", ylabel=f"{label} gradient RMSE")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Squared-loss gradient validation ({seeds} seeds; mean pathwise RMSE)")
    figure.savefig(output / "smooth_convergence.png", dpi=160)
    print("Smooth validation passed: forward, reconstruction and both gradients improve.")
    print(f"Maximum per-path adjoint/backprop S0 difference: {max_delta_difference:.3e}")
    for method in ("Adjoint", "Backprop"):
        print(f"  {method} sigma RMSE: {average(method, 10, 'sigma_rmse'):.3e}"
              f" -> {average(method, 1000, 'sigma_rmse'):.3e}")
    print(f"  Reconstruction max error (mean across seeds): "
          f"{average('Adjoint', 10, 'reconstruction_max'):.3e} -> "
          f"{average('Adjoint', 1000, 'reconstruction_max'):.3e}")


def call_comparison(output, paths, steps):
    s0, strike, rate, sigma, maturity = 100.0, 100.0, 0.05, 0.2, 1.0
    reference = black_scholes(s0, strike, rate, sigma, maturity)
    for actual, expected in zip(reference, (10.450583572185565, 0.6368306511756191, 37.52403469169379)):
        assert math.isclose(actual, expected, rel_tol=1e-12)
    dt = maturity / steps
    noise = StepNoise(steps, paths, dt)
    increments = noise.stored()
    terminal = milstein_gbm(increments, dt, s0, rate, sigma)
    price = payoff(terminal, "call", rate, maturity, strike)
    bp = backprop_gradients(increments, dt, s0, rate, sigma, loss="call", strike=strike)
    adj = adjoint_gradients(increments, dt, terminal,
                            payoff_derivative(terminal, "call", rate, maturity, strike), rate, sigma)
    regenerated_terminal = milstein_gbm(noise, dt, s0, rate, sigma)
    assert torch.equal(terminal, regenerated_terminal)
    regenerated_adj = adjoint_gradients(
        noise, dt, regenerated_terminal,
        payoff_derivative(regenerated_terminal, "call", rate, maturity, strike), rate, sigma
    )
    for stored_gradient, regenerated_gradient in zip(adj, regenerated_adj):
        assert torch.equal(stored_gradient, regenerated_gradient)
    torch.testing.assert_close(adj[0], bp[0], rtol=1e-11, atol=1e-12)
    # Independently check the analytic delta of the discrete multiplicative solver.
    torch.testing.assert_close(bp[0], payoff_derivative(terminal, "call", rate, maturity, strike)
                               * terminal / s0, rtol=1e-11, atol=1e-12)
    # Cover the payoff kink consistently in both gradient methods.
    at_strike = torch.tensor([strike], dtype=torch.float64, requires_grad=True)
    kink_loss = payoff(at_strike, "call", rate, maturity, strike)
    assert torch.autograd.grad(kink_loss.sum(), at_strike)[0].item() == 0.0
    assert payoff_derivative(at_strike.detach(), "call", rate, maturity, strike).item() == 0.0
    exact, exact_s0, exact_sigma = exact_gbm(increments, dt, s0, rate, sigma)
    exact_weight = payoff_derivative(exact, "call", rate, maturity, strike)
    exact_values = (payoff(exact, "call", rate, maturity, strike),
                    exact_weight * exact_s0, exact_weight * exact_sigma)
    rows = []
    for method, values in (("Black-Scholes", None), ("Exact GBM Monte Carlo", exact_values),
                           ("Backprop", (price, *bp)),
                           ("Adjoint (stored noise)", (price, *adj)),
                           ("Adjoint (regenerated)", (price, *regenerated_adj))):
        for index, quantity in enumerate(("price", "delta", "vega")):
            mean, se = (reference[index], 0.0) if values is None else estimate(values[index])
            difference = mean - reference[index]
            rows.append(dict(method=method, quantity=quantity, estimate=mean, se=se,
                             ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se,
                             reference=reference[index], difference=difference,
                             difference_in_se=difference / se if se else "",
                             paths=paths if values is not None else "", steps=steps if values is not None else "",
                             seed=12345 if values is not None else "", dtype="float64", device="cpu"))
    write_csv(output / "call_greeks.csv", rows)
    print(f"\nCall comparison: N={paths}, L={steps}, seed=12345, float64, CPU")
    print(f"{'Method':24} {'Quantity':8} {'Estimate':>12} {'SE':>12} {'Diff/SE':>10}")
    for row in rows:
        z = row['difference_in_se']
        z_text = f"{z:.3f}" if z != "" else "-"
        print(f"{row['method']:24} {row['quantity']:8} {row['estimate']:12.6f} "
              f"{row['se']:12.6f} {z_text:>10}")
    print(f"Call per-path delta agreement passed; max difference "
          f"{(adj[0] - bp[0]).abs().max().item():.3e}")
    print("Paired Backprop minus exact-GBM Monte Carlo differences (estimate +/- SE):")
    for quantity, numerical, analytical in zip(("price", "delta", "vega"), (price, *bp), exact_values):
        difference, se = estimate(numerical - analytical)
        print(f"  {quantity}: {difference:.6g} +/- {se:.6g}")
    print("Confidence intervals describe sampling error; a miss is not an automatic validation failure.")


def memory_worker(method, paths, steps):
    """One end-to-end evaluation in a fresh process; no warm-up allocations.

    resource reports process peak RSS, including Python/PyTorch and noise, not
    just tensor bytes. Linux reports KiB; macOS reports bytes.
    """
    import resource

    def peak_mib():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024**2 if sys.platform == "darwin" else 1024)

    baseline = peak_mib()
    noise = StepNoise(steps, paths, 1 / steps)
    increments = noise if method == "adjoint-regenerated" else noise.stored()
    if method == "backprop":
        delta, vega = backprop_gradients(increments, 1 / steps, 100.0, 0.05, 0.2, loss="call")
    else:
        terminal = milstein_gbm(increments, 1 / steps, 100.0, 0.05, 0.2)
        weight = payoff_derivative(terminal, "call", 0.05, 1.0, 100.0)
        delta, vega = adjoint_gradients(increments, 1 / steps, terminal, weight, 0.05, 0.2)
    delta_mean, vega_mean = delta.mean().item(), vega.mean().item()
    peak = peak_mib()
    print(json.dumps(dict(method=method, paths=paths, steps=steps, seed=noise.seed,
                          baseline_peak_mib=baseline, peak_rss_mib=peak,
                          increase_over_baseline_mib=max(0.0, peak - baseline),
                          delta=delta_mean, vega=vega_mean, dtype="float64", device="cpu",
                          torch_version=torch.__version__, python_version=sys.version.split()[0],
                          platform=sys.platform)))


def memory_benchmark(output, paths, step_counts):
    if sys.platform not in ("linux", "darwin"):
        raise RuntimeError("CPU peak-RSS benchmark requires Linux/WSL or macOS (resource.ru_maxrss)")
    methods = ("backprop", "adjoint-stored", "adjoint-regenerated")
    rows = []
    for steps in step_counts:
        group = []
        for method in methods:
            # Each worker starts from imports, then allocates its own noise and
            # performs both passes. No other configuration's tensors enter it.
            command = [sys.executable, str(Path(__file__).resolve()), "--memory-worker", method,
                       "--paths", str(paths), "--steps", str(steps)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            row = json.loads(result.stdout)
            group.append(row)
            rows.append(row)
            print(f"{method:22} L={steps:5d} peak={row['peak_rss_mib']:.1f} MiB "
                  f"(above startup peak: {row['increase_over_baseline_mib']:.1f} MiB)", flush=True)
            write_csv(output / "memory_scaling.csv", rows)
        assert group[1]["delta"] == group[2]["delta"]
        assert group[1]["vega"] == group[2]["vega"]
        assert math.isclose(group[0]["delta"], group[1]["delta"], rel_tol=1e-11, abs_tol=1e-12)

    # Import plotting only in the parent after collecting the worker results.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, key, label in zip(figure.subplots(1, 2),
                                ("peak_rss_mib", "increase_over_baseline_mib"),
                                ("Process peak RSS (MiB)", "Increase over startup peak (MiB)")):
        for method in methods:
            selected = [row for row in rows if row["method"] == method]
            axis.semilogx([row["steps"] for row in selected], [row[key] for row in selected],
                         "o-", label=method)
        axis.set(xlabel="Time steps L", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"CPU memory scaling: N={paths}, float64; fresh process per configuration")
    figure.savefig(output / "memory_scaling.png", dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=100_000, help="Call Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=100, help="Call time steps")
    parser.add_argument("--seeds", type=int, default=8, help="Smooth-validation seeds")
    parser.add_argument("--smooth-paths", type=int, default=1000)
    parser.add_argument("--benchmark-memory", action="store_true", help="Run only noise validation and fresh-process CPU memory measurements")
    parser.add_argument("--memory-paths", type=int, default=256, help="Fixed path count for memory sweep")
    parser.add_argument("--memory-steps", type=int, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--memory-worker", choices=("backprop", "adjoint-stored", "adjoint-regenerated"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if min(args.paths, args.smooth_paths, args.memory_paths) < 2 or min(args.steps, args.seeds, *args.memory_steps) < 1:
        parser.error("Path counts must be >= 2; steps and seeds must be >= 1")
    torch.set_num_threads(1)
    if args.memory_worker:
        memory_worker(args.memory_worker, args.paths, args.steps)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    validate_noise()
    if args.benchmark_memory:
        memory_benchmark(args.output, args.memory_paths, args.memory_steps)
    else:
        smooth_validation(args.output, args.seeds, args.smooth_paths)
        call_comparison(args.output, args.paths, args.steps)
    print(f"\nSaved CSV tables and convergence plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
