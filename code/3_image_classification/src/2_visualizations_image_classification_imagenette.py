import argparse
from typing import Dict, List, Tuple

import numpy as np

PALETTE = [
  "#dadbc5",
  "#a7cda4",
  "#56b081",
  "#798ad1",
  "#669abf",
  "#272633", 
  "#4f567d",  
]

def _discover_models(df) -> List[str]:
    prefixes = []
    for c in df.columns:
        if c.endswith("_latency_sec"):
            prefixes.append(c[: -len("_latency_sec")])
    seen = set()
    ordered = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _extract_metrics(df, prefixes: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    lat_by = {}
    corr_by = {}
    for p in prefixes:
        lat_col = f"{p}_latency_sec"
        corr_col = f"{p}_correct"
        if lat_col not in df.columns:
            continue
        lat = df[lat_col].to_numpy(dtype=np.float64)
        lat_by[p] = lat

        if corr_col in df.columns:
            corr = df[corr_col].to_numpy(dtype=np.float64)
        else:
            corr = np.full_like(lat, np.nan, dtype=np.float64)
        corr_by[p] = corr
    return lat_by, corr_by


def plot_latency_boxplot(lat_by: Dict[str, np.ndarray], path: str, yscale: str, title: str):
    import matplotlib.pyplot as plt

    names = list(lat_by.keys())
    data = [lat_by[n][np.isfinite(lat_by[n])] for n in names]

    fig = plt.figure(figsize=(max(10, 2 + 2.2 * len(names)), 5))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(
        data,
        labels=[
            f"{n}\n(med={np.median(d):.3f}s)" if len(d) else f"{n}\n(med=n/a)"
            for n, d in zip(names, data)
        ],
        showmeans=True,
        patch_artist=True,
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.75)

    ax.set_ylabel("Latency (seconds)", fontsize=10)
    ax.set_yscale(yscale)
    ax.set_xticklabels(names, rotation=15, ha="right")

    if yscale == "log":
        all_vals = np.concatenate([d[d > 0] for d in data if len(d)], axis=0) if any(len(d) for d in data) else None
        if all_vals is not None and all_vals.size > 0:
            ax.set_ylim(bottom=max(float(np.min(all_vals)) * 0.8, 1e-4))

    ax.set_title(title, fontsize=14)

    plt.setp(ax.get_xticklabels(), size=5)
    fig.tight_layout()
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=9.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_accuracy_bar(corr_by: Dict[str, np.ndarray], path: str, title: str):
    import matplotlib.pyplot as plt

    names = list(corr_by.keys())
    accs = []
    for n in names:
        v = corr_by[n]
        v = v[np.isfinite(v)]
        accs.append(float(np.mean(v)) if v.size else np.nan)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    ax.bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor="black", linewidth=1.0)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=9.5)
    ax.set_ylim(0, 1.0)

    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cumulative_latency(lat_by: Dict[str, np.ndarray], path: str, title: str, yscale: str = "linear"):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    for i, (name, lat) in enumerate(lat_by.items()):
        lat = np.asarray(lat, dtype=np.float64)
        lat = np.where(np.isfinite(lat), lat, 0.0)
        cum = np.cumsum(lat)
        x = np.arange(1, len(cum) + 1)

        ax.plot(x, cum, label=name, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Request #", fontsize=20)
    ax.set_ylabel("Cumul. latency (seconds)", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.set_yscale(yscale)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_latency_cdf(lat_by: Dict[str, np.ndarray], path: str, title: str, xscale: str = "log"):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    for i, (name, lat) in enumerate(lat_by.items()):
        lat = np.asarray(lat, dtype=np.float64)
        lat = lat[np.isfinite(lat)]
        lat = lat[lat > 0]
        if lat.size == 0:
            continue

        lat_sorted = np.sort(lat)
        y = np.arange(1, lat_sorted.size + 1) / lat_sorted.size

        ax.plot(lat_sorted, y, label=name, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Latency (seconds)", fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xscale(xscale)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_ylim(0, 1.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", required=False, default="imgcls_benchmark_log_3925_validation.csv", help="Path to CSV produced by the benchmark script.")
    ap.add_argument("--latency_scale", default="log", choices=["linear", "log"], help="Y-scale for latency boxplot.")
    ap.add_argument("--cum_latency_scale", default="linear", choices=["linear", "log"], help="Y-scale for cumulative plot.")
    ap.add_argument("--latency_boxplot_path", default="ic_latency_boxplot.png")
    ap.add_argument("--acc_bar_path", default="ic_accuracy_bar.png")
    ap.add_argument("--cum_latency_path", default="ic_cumulative_latency.png")
    ap.add_argument("--title_prefix", default="", help="Optional prefix for plot titles.")
    ap.add_argument("--cdf_path", default="ic_latency_cdf.png")
    ap.add_argument("--cdf_xscale", default="log", choices=["linear", "log"])
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.log_csv)

    prefixes = _discover_models(df)
    if not prefixes:
        raise RuntimeError("No model latency columns found (expected columns ending with '_latency_sec').")

    lat_by, corr_by = _extract_metrics(df, prefixes)

    n = len(df)
    title_prefix = (args.title_prefix.strip() + " ") if args.title_prefix.strip() else ""

    plot_latency_boxplot(
        lat_by,
        args.latency_boxplot_path,
        yscale=args.latency_scale,
        title=f"{title_prefix}Lat. boxplot (n={n}, frgfm/imagenette.val)",
    )
    print(f"Saved: {args.latency_boxplot_path}")

    plot_accuracy_bar(
        corr_by,
        args.acc_bar_path,
        title=f"{title_prefix}Accuracy (n={n}, frgfm/imagenette.val)",
    )
    print(f"Saved: {args.acc_bar_path}")

    plot_cumulative_latency(
        lat_by,
        args.cum_latency_path,
        title=f"{title_prefix}Cumul. lat. vs request #",
        yscale=args.cum_latency_scale,
    )
    print(f"Saved: {args.cum_latency_path}")

    plot_latency_cdf(
        lat_by,
        args.cdf_path,
        title=f"{title_prefix}Latency CDF (n={n}, val, frgfm/imagenette)",
        xscale=args.cdf_xscale,
    )
    print(f"Saved: {args.cdf_path}")


if __name__ == "__main__":
    main()