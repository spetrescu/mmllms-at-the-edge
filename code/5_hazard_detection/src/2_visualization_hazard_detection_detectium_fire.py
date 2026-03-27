import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = [
  "#1c1c23",
  "#297072",
  "#c5b8b8",
]

METHOD_COLORS = {
    "yolo": PALETTE[0],
    "gemma_full": PALETTE[1],
    "periodic_k5": PALETTE[2],
}

METHOD_LABELS = {
    "yolo": "yolov8",
    "gemma_full": "gemma3n-e2b",
    "periodic_k5": "gemma3n-e2b-periodic-k5",
}

METHOD_MARKERS = {
    "yolo": "o",
    "gemma_full": "s",
    "periodic_k5": "D",
}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_by_scenario(summary: pd.DataFrame, out_path: str) -> None:
    scenarios = ["fire_videos", "nonfire_videos"]
    methods = ["yolo", "gemma_full", "periodic_k5"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(scenarios))
    width = 0.25

    for i, m in enumerate(methods):
        ys = []
        for sc in scenarios:
            row = summary[(summary["scenario"] == sc) & (summary["method"] == m)]
            ys.append(float(row["accuracy"].iloc[0]) if len(row) else np.nan)

        ax.bar(
            x + (i - 1) * width,
            ys,
            width,
            label=METHOD_LABELS[m],
            edgecolor="black",
            color=METHOD_COLORS[m],
        )

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fire videos", "No-fire videos"],fontsize=18)
    ax.set_title("Fire vs no-fire accuracy",fontsize=18)

    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='lower right')
    save_fig(out_path)


def plot_accuracy_combined(summary: pd.DataFrame, out_path: str) -> None:
    methods = ["yolo", "gemma_full", "periodic_k5"]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    vals = []
    for m in methods:
        row = summary[(summary["scenario"] == "combined") & (summary["method"] == m)]
        vals.append(float(row["accuracy"].iloc[0]) if len(row) else np.nan)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, edgecolor="black", color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (combined across fire + no-fire)")
    save_fig(out_path)


def plot_latency_cdf(samples: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    series = [
        ("yolo", METHOD_LABELS["yolo"], samples["lat_yolo_ms"].dropna().values),
        ("gemma_full", METHOD_LABELS["gemma_full"], samples["lat_gemma_full_ms"].dropna().values),
        ("periodic_k5", METHOD_LABELS["periodic_k5"], samples["lat_periodic_ms"].dropna().values),
    ]

    for method_key, label, arr in series:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        xs = np.sort(arr)
        ys = np.linspace(0, 1, xs.size)
        ax.plot(xs, ys, label=label, linewidth=1.5, color=METHOD_COLORS[method_key])

    ax.set_xlabel("Latency per sample (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF (all samples)")
    ax.legend()
    save_fig(out_path)


def plot_latency_boxplot(samples: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    data = [
        samples["lat_yolo_ms"].dropna().values,
        samples["lat_gemma_full_ms"].dropna().values,
        samples["lat_periodic_ms"].dropna().values,
    ]
    labels = [METHOD_LABELS["yolo"], METHOD_LABELS["gemma_full"], METHOD_LABELS["periodic_k5"]]

    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    box_colors = [METHOD_COLORS["yolo"], METHOD_COLORS["gemma_full"], METHOD_COLORS["periodic_k5"]]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)

    ax.set_ylabel("Latency per sample (ms)")
    ax.set_title("Latency boxplot (all samples)")
    save_fig(out_path)


def plot_tradeoff_cumlat_vs_accuracy(summary: pd.DataFrame, out_path: str) -> None:
    METHOD_MARKERS = {
        "yolo": "o",
        "gemma_full": "P",
        "periodic_k5": "D",
    }

    df = summary[summary["scenario"] == "combined"].copy()

    fig, ax = plt.subplots(figsize=(5, 4))

    for _, r in df.iterrows():
        method_key = str(r["method"])

        ax.scatter(
            r["cumulative_latency_s"],
            r["accuracy"],
            marker=METHOD_MARKERS.get(method_key, "o"),
            s=100,
            facecolor=METHOD_COLORS.get(method_key),
            edgecolor="black",
            linewidth=1,
            label=METHOD_LABELS.get(method_key, method_key),
            zorder=3,
        )

    ax.grid(True, linestyle="--", alpha=0.2)

    ax.set_xlabel("Cumul. lat. to process all samples (s)", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Tradeoff: cumulative latency vs accuracy", fontsize=15)

    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right", frameon=True)

    save_fig(out_path)

def main():
    out_dir = "output"
    tables_dir = os.path.join(out_dir, "tables")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    samples_path = os.path.join(tables_dir, "samples.csv")
    summary_path = os.path.join(tables_dir, "summary.csv")

    samples = pd.read_csv(samples_path)
    summary = pd.read_csv(summary_path)

    if len(summary):
        plot_accuracy_by_scenario(summary, os.path.join(plots_dir, "accuracy_by_scenario.png"))
        plot_tradeoff_cumlat_vs_accuracy(summary, os.path.join(plots_dir, "tradeoff_cumlat_vs_accuracy.png"))

    print(f"Plots at {plots_dir}")

if __name__ == "__main__":
    main()