"""Presentation memory figures. Run with --help; Linux/WSL, CPU float64 only."""

import argparse
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

from gbm_adjoint import (
    HARDWARE_CAPTION, StepNoise, adjoint_gradients, evaluate_call, milstein_gbm,
    payoff, payoff_derivative, validate_noise, write_csv,
)

METHODS = ("backprop", "adjoint-stored", "adjoint-regenerated")
LABELS = ("Backprop", "Adjoint: stored noise", "Adjoint: regenerated noise")
COLORS = ("#2878b5", "#e68613", "#32954b")
# Edit the fill opacity here; outline curves remain fully opaque.
FILL_ALPHA = 0.18


def peak_mib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(method, paths, steps):
    baseline = peak_mib()
    delta, vega = evaluate_call(method, paths, steps)
    peak = peak_mib()
    return dict(method=method, paths=paths, steps=steps, seed=12345,
                baseline_peak_mib=baseline, peak_rss_mib=peak,
                increase_mib=max(0.0, peak - baseline), delta=delta, vega=vega,
                dtype="float64", device="cpu", threads=1, launch="fresh fork",
                torch_version=torch.__version__, python_version=sys.version.split()[0],
                platform=sys.platform, hardware=HARDWARE_CAPTION)


def child_job(connection, function, arguments):
    try:
        connection.send(("result", function(*arguments)))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def fresh_measurement(context, method, paths, steps):
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=child_job, args=(send, measure, (method, paths, steps)))
    process.start()
    send.close()
    try:
        kind, result = receive.recv()
    finally:
        receive.close()
        process.join()
    if process.exitcode or kind != "result":
        raise RuntimeError(f"Memory worker failed: {result}")
    return result


def check_greeks(group):
    values = {row["method"]: (row["delta"], row["vega"]) for row in group}
    assert values["adjoint-stored"] == values["adjoint-regenerated"]
    assert math.isclose(values["backprop"][0], values["adjoint-stored"][0],
                        rel_tol=1e-11, abs_tol=1e-12)


def collect_memory(context, output, repetitions, fixed_paths, fixed_steps, steps, paths):
    rows, summaries = [], []
    configurations = [("steps", fixed_paths, count) for count in steps]
    configurations += [("paths", count, fixed_steps) for count in paths]
    order_rng = random.Random(2020)
    for sweep, path_count, step_count in configurations:
        group = []
        for repetition in range(1, repetitions + 1):
            order = list(METHODS)
            order_rng.shuffle(order)
            trial = []
            for position, method in enumerate(order, 1):
                row = fresh_measurement(context, method, path_count, step_count)
                row.update(sweep=sweep, repetition=repetition, order=position, order_seed=2020)
                trial.append(row)
            check_greeks(trial)
            group.extend(trial)
        rows.extend(group)
        for method in METHODS:
            selected = [row for row in group if row["method"] == method]
            assert len(selected) == repetitions
            assert len({(row["delta"], row["vega"]) for row in selected}) == 1
            summary = dict(sweep=sweep, method=method, paths=path_count, steps=step_count,
                           repetitions=repetitions)
            for key in ("peak_rss_mib", "increase_mib"):
                samples = [row[key] for row in selected]
                q25, _, q75 = statistics.quantiles(samples, n=4, method="inclusive")
                summary.update({f"{key}_median": statistics.median(samples),
                                f"{key}_q25": q25, f"{key}_q75": q75})
            summaries.append(summary)
        write_csv(output / "memory_repeated_samples.csv", rows)
        write_csv(output / "memory_repeated_summary.csv", summaries)
        text = ", ".join(f"{LABELS[METHODS.index(row['method'])]} {row['peak_rss_mib_median']:.1f} MiB"
                         for row in summaries[-3:])
        print(f"N={path_count:,}, L={step_count:,}; {repetitions} fresh workers each: {text}", flush=True)
    return summaries


def profile_evaluation(connection, method, paths, steps):
    def phase(name):
        connection.send(("phase", dict(phase=name, seconds=time.perf_counter())))

    phase("ready")
    connection.recv()  # Allow the parent to record baseline RSS before work starts.
    phase("noise")
    noise = StepNoise(steps, paths, 1 / steps)
    increments = noise if method == "adjoint-regenerated" else noise.stored()
    phase("forward")
    if method == "backprop":
        spot = torch.full((paths,), 100.0, dtype=torch.float64, requires_grad=True)
        volatility = torch.full_like(spot, 0.2, requires_grad=True)
        terminal = milstein_gbm(increments, 1 / steps, spot, 0.05, volatility)
        loss = payoff(terminal, "call", 0.05, 1.0, 100.0)
        phase("backward")
        gradients = torch.autograd.grad(loss.sum(), (spot, volatility))
    else:
        terminal = milstein_gbm(increments, 1 / steps, 100.0, 0.05, 0.2)
        weight = payoff_derivative(terminal, "call", 0.05, 1.0, 100.0)
        phase("backward")
        gradients = adjoint_gradients(increments, 1 / steps, terminal, weight, 0.05, 0.2)
    result = dict(method=method, delta=gradients[0].mean().item(), vega=gradients[1].mean().item())
    phase("finish")
    return result


def profile_job(connection, method, paths, steps):
    try:
        result = profile_evaluation(connection, method, paths, steps)
        connection.send(("result", result))
        connection.recv()  # Keep the worker alive for the final RSS sample.
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def profile_memory(context, output, paths, steps):
    samples, phases, results = [], [], []
    page_mib = os.sysconf("SC_PAGE_SIZE") / 1024**2
    for method in METHODS:
        receive, send = context.Pipe()
        process = context.Process(target=profile_job, args=(send, method, paths, steps))
        process.start()
        send.close()
        kind, ready = receive.recv()
        assert kind == "phase" and ready["phase"] == "ready"
        start = time.perf_counter()
        statm = Path(f"/proc/{process.pid}/statm")

        def sample():
            # External sampling avoids a sampler thread competing with PyTorch's GIL.
            rss = int(statm.read_text().split()[1]) * page_mib
            samples.append(dict(method=method, paths=paths, steps=steps,
                                seconds=time.perf_counter() - start, rss_mib=rss))

        sample()
        receive.send("start")
        try:
            done = False
            while not done:
                sample()
                while receive.poll():
                    kind, payload = receive.recv()
                    if kind == "phase":
                        phases.append(dict(method=method, phase=payload["phase"],
                                           seconds=payload["seconds"] - start))
                    elif kind == "result":
                        results.append(payload)
                        sample()
                        receive.send("stop")
                        done = True
                        break
                    else:
                        raise RuntimeError(payload)
                if not done:
                    time.sleep(0.002)
        finally:
            receive.close()
            process.join()
        assert process.exitcode == 0
    check_greeks(results)
    write_csv(output / "memory_timeline_samples.csv", samples)
    write_csv(output / "memory_timeline_phases.csv", phases)
    return samples, phases


def new_figure(width=11, height=4.8):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save_figure(figure, output, name):
    figure.supxlabel(HARDWARE_CAPTION, fontsize=8, color="0.4")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=200)


def plot_scaling(output, summaries, fixed_paths, fixed_steps, repetitions):
    for key, ylabel, name in (
        ("peak_rss_mib", "Peak process RAM (MiB)", "memory_scaling_presentation"),
        ("increase_mib", "Peak RAM above worker baseline (MiB)", "memory_above_baseline"),
    ):
        figure = new_figure()
        for axis, sweep, xlabel, title in zip(
            figure.subplots(1, 2), ("steps", "paths"), ("Time steps L", "Simulated paths N"),
            (f"Step scaling: N = {fixed_paths:,}", f"Path scaling: L = {fixed_steps:,}")
        ):
            for method, label, color in zip(METHODS, LABELS, COLORS):
                selected = sorted((r for r in summaries if r["sweep"] == sweep and r["method"] == method),
                                  key=lambda row: row[sweep])
                x = [r[sweep] for r in selected]
                axis.semilogx(x, [r[f"{key}_median"] for r in selected], "o-", color=color, label=label)
                axis.fill_between(x, [r[f"{key}_q25"] for r in selected],
                                  [r[f"{key}_q75"] for r in selected], color=color, alpha=0.18)
            axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=9)
        figure.suptitle(f"Memory scaling | Median and middle 50% of {repetitions} fresh-process runs")
        save_figure(figure, output, name)


def plot_timeline(output, samples, phases, paths, steps, name="memory_timeline"):
    figure = new_figure(12, 4.5)
    top = max(row["rss_mib"] for row in samples) * 1.12
    bottom = max(0, min(row["rss_mib"] for row in samples) - 15)
    for axis, method, label, color in zip(figure.subplots(1, 3), METHODS, LABELS, COLORS):
        selected = [row for row in samples if row["method"] == method]
        markers = [row for row in phases if row["method"] == method]
        axis.plot([row["seconds"] for row in selected], [row["rss_mib"] for row in selected], color=color, linewidth=2)
        axis.fill_between([row["seconds"] for row in selected], bottom,
                          [row["rss_mib"] for row in selected], color=color, alpha=FILL_ALPHA)
        for first, last, shade in zip(markers, markers[1:], ("#ececec", "#dcebf4", "#fae5cf")):
            axis.axvspan(first["seconds"], last["seconds"], color=shade, alpha=0.2, zorder=0)
        axis.set(xlabel="Elapsed time (s)", ylabel="Process RAM (MiB)", title=label,
                 ylim=(bottom, top), xlim=(0, selected[-1]["seconds"]))
        axis.grid(alpha=0.2)
    from matplotlib.patches import Patch
    figure.axes[0].legend(handles=[Patch(color=color, label=label) for color, label in
                                  zip(("#ececec", "#dcebf4", "#fae5cf"), ("Noise setup", "Forward", "Backward"))],
                         loc="upper left", fontsize=8)
    figure.suptitle(f"RAM over one evaluation | N = {paths:,}, L = {steps:,} | External sampling ≈ 2 ms")
    save_figure(figure, output, name)


def plot_tradeoff(output, summaries, paths, steps):
    # Match both dimensions and methods; never join merely on method name.
    with (output / "runtime_summary.csv").open() as handle:
        runtime = list(csv.DictReader(handle))
    figure = new_figure(8, 5)
    axis = figure.subplots()
    for method, label, color in zip(METHODS, LABELS, COLORS):
        memory_rows = [r for r in summaries if r["sweep"] == "paths" and r["method"] == method
                       and r["paths"] == paths and r["steps"] == steps]
        timing_rows = [r for r in runtime if r["sweep"] == "paths" and r["method"] == method
                       and int(r["paths"]) == paths and int(r["steps"]) == steps]
        if len(memory_rows) != 1 or len(timing_rows) != 1:
            raise ValueError(f"Need one matching memory/runtime configuration for {method}, N={paths}, L={steps}")
        memory, timing = memory_rows[0], timing_rows[0]
        x, y = memory["peak_rss_mib_median"], 1000 * float(timing["median_seconds"])
        axis.errorbar(x, y, xerr=[[x - memory["peak_rss_mib_q25"]], [memory["peak_rss_mib_q75"] - x]],
                      yerr=[[y - 1000 * float(timing["q25_seconds"])],
                            [1000 * float(timing["q75_seconds"]) - y]],
                      fmt="o", color=color, markersize=9, capsize=4, label=label)
    axis.set(xlabel="Median peak process RAM (MiB)", ylabel="Median runtime (ms)",
             title=f"Speed–memory tradeoff | N = {paths:,}, L = {steps:,}")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.suptitle("Lower and further left is better | Bars show the middle 50%")
    save_figure(figure, output, "memory_runtime_tradeoff")


def plot_filled_timeline(output, samples, phases, paths, steps):
    """Overlay measured RSS curves with translucent fills and the existing plot style."""
    figure = new_figure(11, 4.5)
    axis = figure.subplots()
    bottom = max(0, min(row["rss_mib"] for row in samples) - 15)
    top = max(row["rss_mib"] for row in samples) * 1.12
    for method, label, color in zip(METHODS, LABELS, COLORS):
        selected = [row for row in samples if row["method"] == method]
        times = [row["seconds"] for row in selected]
        values = [row["rss_mib"] for row in selected]
        axis.fill_between(times, bottom, values, color=color, alpha=FILL_ALPHA)
        axis.plot(times, values, color=color, linewidth=2, label=label)
    axis.set(xlabel="Elapsed time (s)", ylabel="Process RAM (MiB)",
             ylim=(bottom, top), xlim=(0, max(row["seconds"] for row in samples)))
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    figure.suptitle(f"RAM over one evaluation | N = {paths:,}, L = {steps:,} | External sampling ≈ 2 ms")
    save_figure(figure, output, "memory_utilization")
    plot_timeline(output, samples, phases, paths, steps, name="memory_utilization_comparison")


def read_measurements(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key in ("paths", "steps", "repetitions"):
                row[key] = int(row[key])
            elif key in ("seconds", "rss_mib") or key.endswith(("_median", "_q25", "_q75")):
                row[key] = float(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--step-paths", type=int, default=256)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--path-steps", type=int, default=100)
    parser.add_argument("--paths", type=int, nargs="+", default=[1000, 10000, 30000, 50000])
    parser.add_argument("--profile-paths", type=int, default=2000)
    parser.add_argument("--profile-steps", type=int, default=1000)
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from saved CSV measurements")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if not args.plot_only and sys.platform != "linux":
        parser.error("This external-RSS profiler requires Linux/WSL (/proc and fresh fork workers)")
    if min(args.repetitions, args.step_paths, args.profile_paths, *args.paths) < 2 or min(
        args.path_steps, args.profile_steps, *args.steps
    ) < 1:
        parser.error("Repetitions/path counts must be >= 2; steps must be positive")
    if not (args.output / "runtime_summary.csv").exists():
        parser.error("Run gbm_adjoint.py --benchmark-runtime first; its summary is needed for the tradeoff plot")
    with (args.output / "runtime_summary.csv").open() as handle:
        matching = [row for row in csv.DictReader(handle) if row["sweep"] == "paths"
                    and int(row["paths"]) == max(args.paths) and int(row["steps"]) == args.path_steps]
    if sorted(row["method"] for row in matching) != sorted(METHODS):
        parser.error("The runtime summary must contain all three methods at the largest path count and --path-steps")
    if args.plot_only:
        summaries = read_measurements(args.output / "memory_repeated_summary.csv")
        samples = read_measurements(args.output / "memory_timeline_samples.csv")
        phases = read_measurements(args.output / "memory_timeline_phases.csv")
        if any(row["repetitions"] != args.repetitions for row in summaries):
            parser.error("--repetitions must match the saved summaries when using --plot-only")
    else:
        torch.set_num_threads(1)
        validate_noise()
        args.output.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("fork")
        summaries = collect_memory(context, args.output, args.repetitions, args.step_paths,
                                   args.path_steps, args.steps, args.paths)
        # Keep plotting imports/allocations out of the parent until all workers finish.
        samples, phases = profile_memory(context, args.output, args.profile_paths, args.profile_steps)
    plot_scaling(args.output, summaries, args.step_paths, args.path_steps, args.repetitions)
    plot_timeline(args.output, samples, phases, args.profile_paths, args.profile_steps)
    plot_filled_timeline(args.output, samples, phases, args.profile_paths, args.profile_steps)
    plot_tradeoff(args.output, summaries, max(args.paths), args.path_steps)
    print(f"Saved presentation figures and measurements to {args.output.resolve()}")


if __name__ == "__main__":
    main()
