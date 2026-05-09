import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_results(csv_path, distances, decoders):
    data = {d: {dec: [] for dec in decoders} for d in distances}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                d = int(row["d"])
            except (KeyError, ValueError):
                continue
            decoder = row.get("decoder")
            if d not in data or decoder not in decoders:
                continue
            try:
                p = float(row["p"])
                p_l = float(row["p_L"])
            except (KeyError, ValueError):
                continue
            data[d][decoder].append((p, p_l))

    for d in distances:
        for decoder in decoders:
            data[d][decoder].sort(key=lambda item: item[0])

    return data


def find_crossover(points_a, points_b):
    if len(points_a) < 2 or len(points_b) < 2:
        return None

    for (p1, a1), (p2, a2), (_, b1), (_, b2) in zip(
        points_a[:-1], points_a[1:], points_b[:-1], points_b[1:]
    ):
        diff1 = a1 - b1
        diff2 = a2 - b2
        if diff1 == 0:
            return p1, a1
        if diff1 * diff2 < 0:
            log_p1 = math.log10(p1)
            log_p2 = math.log10(p2)
            log_a1 = math.log10(a1)
            log_a2 = math.log10(a2)
            log_b1 = math.log10(b1)
            log_b2 = math.log10(b2)
            log_diff1 = log_a1 - log_b1
            log_diff2 = log_a2 - log_b2
            t = log_diff1 / (log_diff1 - log_diff2)
            log_p = log_p1 + t * (log_p2 - log_p1)
            log_y = log_a1 + t * (log_a2 - log_a1)
            return 10 ** log_p, 10 ** log_y

    return None


def compute_lambda_series(data, distances, decoder, pairs):
    series = {}
    for d1, d2 in pairs:
        points_1 = {p: l for p, l in data[d1][decoder] if l > 0}
        points_2 = {p: l for p, l in data[d2][decoder] if l > 0}
        shared = sorted(set(points_1) & set(points_2))
        lambdas = []
        for p in shared:
            l1 = points_1[p]
            l2 = points_2[p]
            if l2 == 0:
                continue
            lambdas.append((p, l1 / l2))
        series[(d1, d2)] = lambdas
    return series


def plot_supplementary(data, output_path):
    distances = sorted(data.keys())
    colors = plt.cm.tab10.colors
    pairs = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    distance_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=2, label=f"d={d}")
        for idx, d in enumerate(distances)
    ]

    fig_gnn, gnn_ax = plt.subplots(figsize=(11, 8.5))
    for idx, d in enumerate(distances):
        gnn_points = [(p, l) for p, l in data[d]["GraphSAGE"] if l > 0]
        if not gnn_points:
            continue
        color = colors[idx % len(colors)]
        gnn_p, gnn_l = zip(*gnn_points)
        gnn_ax.loglog(
            gnn_p,
            gnn_l,
            color=color,
            linewidth=2.0,
            label=f"d={d}",
        )

    gnn_ax.set_title("GraphSAGE", fontsize=16)
    gnn_ax.set_xlabel("Physical error rate $p$")
    gnn_ax.set_ylabel("Logical error rate $p_L$")
    gnn_ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    gnn_ax.legend(
        handles=distance_handles,
        loc="upper left",
        frameon=False,
        title="Distance",
    )
    fig_gnn.tight_layout()
    fig_gnn.savefig(output_path.with_name(f"{output_path.stem}_graphsage{output_path.suffix}"), dpi=300)
    plt.close(fig_gnn)

    fig_mwpm, mwpm_ax = plt.subplots(figsize=(11, 8.5))
    for idx, d in enumerate(distances):
        mwpm_points = [(p, l) for p, l in data[d]["MWPM"] if l > 0]
        if not mwpm_points:
            continue
        color = colors[idx % len(colors)]
        mwpm_p, mwpm_l = zip(*mwpm_points)
        mwpm_ax.loglog(
            mwpm_p,
            mwpm_l,
            color=color,
            linewidth=2.0,
            label=f"d={d}",
        )

    mwpm_ax.set_title("MWPM", fontsize=16)
    mwpm_ax.set_xlabel("Physical error rate $p$")
    mwpm_ax.set_ylabel("Logical error rate $p_L$")
    mwpm_ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    mwpm_ax.legend(
        handles=distance_handles,
        loc="upper left",
        frameon=False,
        title="Distance",
    )
    fig_mwpm.tight_layout()
    fig_mwpm.savefig(output_path.with_name(f"{output_path.stem}_mwpm{output_path.suffix}"), dpi=300)
    plt.close(fig_mwpm)

    gnn_lambda = compute_lambda_series(data, distances, "GraphSAGE", pairs)
    mwpm_lambda = compute_lambda_series(data, distances, "MWPM", pairs)

    fig_lambda, lambda_ax = plt.subplots(figsize=(11, 8.5))
    pair_colors = plt.cm.tab20.colors
    for idx, pair in enumerate(pairs):
        color = pair_colors[idx % len(pair_colors)]
        gnn_points = gnn_lambda.get(pair, [])
        mwpm_points = mwpm_lambda.get(pair, [])

        if gnn_points:
            p_vals, lambda_vals = zip(*gnn_points)
            lambda_ax.semilogx(
                p_vals,
                lambda_vals,
                color=color,
                linewidth=2.0,
                label=f"d{pair[0]}-d{pair[1]} GraphSAGE",
            )

        if mwpm_points:
            p_vals, lambda_vals = zip(*mwpm_points)
            lambda_ax.semilogx(
                p_vals,
                lambda_vals,
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"d{pair[0]}-d{pair[1]} MWPM",
            )

    lambda_ax.axhline(1.0, color="black", linewidth=1.0, linestyle=":")
    lambda_ax.set_title("Lambda between distances ($\\Lambda = p_L(d_1) / p_L(d_2)$)", fontsize=14)
    lambda_ax.set_xlabel("Physical error rate $p$")
    lambda_ax.set_ylabel("$\\Lambda$")
    lambda_ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    lambda_ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    caption = (
        "GraphSAGE shows $\\Lambda < 1$ across tested $p$; MWPM shows $\\Lambda > 1$ below threshold.\n"
        "Closing the gap requires soft information + circuit-level training (see future work)."
    )
    fig_lambda.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    fig_lambda.tight_layout(rect=(0, 0.06, 1, 1))
    fig_lambda.savefig(output_path.with_name(f"{output_path.stem}_lambda{output_path.suffix}"), dpi=300)
    plt.close(fig_lambda)


def main():
    parser = argparse.ArgumentParser(description="Generate supplementary Lambda scaling plot.")
    parser.add_argument(
        "--input",
        default="code/results/nn_vs_gnn_comparison/specialist_sweep_results.csv",
        help="Path to the sweep results CSV.",
    )
    parser.add_argument(
        "--output",
        default="supplementary_lambda_scaling.pdf",
        help="Path to the output PDF.",
    )
    args = parser.parse_args()

    distances = [3, 5, 7, 9, 11, 13]
    decoders = ["GraphSAGE", "MWPM"]

    data = load_results(args.input, distances, decoders)
    plot_supplementary(data, Path(args.output))


if __name__ == "__main__":
    main()
