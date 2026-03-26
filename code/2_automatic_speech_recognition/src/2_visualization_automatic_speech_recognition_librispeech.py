import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_PALETTE = [
  "#76716f", 
  "#9c6c6f", 
  "#bf646f", 
  "#e0586f", 
  "#e8788a", 
  "#ec98a7"
]

VOSK_PALETTE = ["#7cb179", "#468051"]

GEMMA_PALETTE = {
    "gemma3n:e2b": "#F54927",
    "gemma3n:e4b": "#4e9fed",
    "gemma3n_e2b": "#b08dc7",
    "gemma3n_e4b": "#4e9fed",
}

def find_model_prefixes(df: pd.DataFrame) -> List[str]:
    prefixes = []
    for c in df.columns:
        if c.endswith("_latency_sec"):
            prefixes.append(c[: -len("_latency_sec")])
    return sorted(prefixes)


def pretty_label(prefix: str) -> str:
    if prefix.startswith("whisper"):
        return "whisper:" + prefix[len("Whisper_") :]

    return prefix


def model_sort_key(label: str) -> Tuple[int, int, str]:
    if label.startswith("whisper:"):
        order = {
            "tiny": 5,
            "base": 4,
            "small": 3,
            "medium": 2,
            "large-v3-turbo": 1,
            "large-v3": 0,
        }
        name = label.split(":", 1)[1]
        return (0, order.get(name, 999), label)

    if label.startswith("vosk:"):
        sub = label.split(":", 1)[1]
        return (2, 0 if "small" in sub.lower() else 1, label)

    if label.startswith("gemma3n:"):
        sub = label.split(":", 1)[1]
        return (1, 0 if "e2b" in sub.lower() else 1, label)

    return (3, 999, label)


def get_series(df: pd.DataFrame, prefix: str, suffix: str) -> np.ndarray:
    c = f"{prefix}{suffix}"
    if c not in df.columns:
        return np.array([], dtype=np.float64)
    x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    return x

def whisper_size_name(label: str) -> str:
    return label.split(":", 1)[1].strip() if label.startswith("Whisper:") else ""


def assign_colors(labels: List[str]) -> Dict[str, str]:
    colors: Dict[str, str] = {}

    whisper_color_by_size = {
        "large-v3": BASE_PALETTE[0],
        "large-v3-turbo": BASE_PALETTE[1],
        "medium": BASE_PALETTE[2],
        "small": BASE_PALETTE[3],
        "base": BASE_PALETTE[4],
        "tiny": BASE_PALETTE[5],
    }

    vosk_cycle = iter(VOSK_PALETTE)

    fallback_i = 0

    for lab in labels:
        print(lab)
        if lab.startswith("gemma3n"):
            colors[lab] = GEMMA_PALETTE.get(lab, "#8a4bd6")
            print(colors[lab])
            continue

        if lab.startswith("whisper:"):
            size = whisper_size_name(lab)
            colors[lab] = whisper_color_by_size.get(size, BASE_PALETTE[min(5, len(BASE_PALETTE) - 1)])
            continue

        if lab.startswith("vosk"):
            try:
                colors[lab] = next(vosk_cycle)
            except StopIteration:
                colors[lab] = VOSK_PALETTE[(len([k for k in colors if k.startswith("vosk:")]) - 1) % len(VOSK_PALETTE)]
            continue

        colors[lab] = BASE_PALETTE[fallback_i % len(BASE_PALETTE)]
        fallback_i += 1

    return colors

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_box_colors(bp, colors: List[str]) -> None:
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor("black")
        box.set_linewidth(1.0)

    for k in ["medians", "whiskers", "caps"]:
        for item in bp[k]:
            item.set_color("black")
            item.set_linewidth(1.0)

    for mean in bp.get("means", []):
        mean.set_marker("o")
        mean.set_markerfacecolor("black")
        mean.set_markeredgecolor("black")
        mean.set_markersize(4)


def save_boxplot(
    data_by_label: Dict[str, np.ndarray],
    colors_by_label: Dict[str, str],
    out_path: str,
    title: str,
    ylabel: str,
    yscale: str = "linear",
    y_min: float = None,
    y_max: float = None,
    hline: float = None,
) -> None:
    labels = list(data_by_label.keys())
    data = [data_by_label[l] for l in labels]
    colors = [colors_by_label[l] for l in labels]

    plt.figure(figsize=(max(10, 2 + 2.2 * len(labels)), 5))
    plt.figure(figsize=(6.5, 5))
    bp = plt.boxplot(
        data,
        labels=[
            f"{l}"
            # f"{l}\n(median={np.nanmedian(v):.3f})" if np.isfinite(np.nanmedian(v)) else f"{l}\n(median=n/a)"
            for l, v in zip(labels, data)
        ],
        showmeans=True,
        patch_artist=True,
    )
    apply_box_colors(bp, colors)

    plt.title(title, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.yscale(yscale)

    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)

    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)

    plt.xticks(fontsize=12, rotation=33)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_latency_cdf(
    lat_by_label: Dict[str, np.ndarray],
    colors_by_label: Dict[str, str],
    out_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.figure(figsize=(5, 4))
    for label, lat in lat_by_label.items():
        v = lat[np.isfinite(lat)]
        v = v[v >= 0]
        if v.size == 0:
            continue
        xs = np.sort(v)
        ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
        plt.step(xs, ys, where="post", label=label, color=colors_by_label[label], linewidth=2)

    plt.title(title, fontsize=15)
    plt.xlabel("Latency per request (seconds)")
    plt.ylabel("CDF", fontsize=15)
    plt.ylim(0, 1.0)
    plt.grid(True, which="both", axis="both", linewidth=0.5, alpha=0.4)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cumulative_time(
    lat_by_label: Dict[str, np.ndarray],
    colors_by_label: Dict[str, str],
    out_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.figure(figsize=(5, 4))
    for label, lat in lat_by_label.items():
        v = lat.copy()
        v[~np.isfinite(v)] = 0.0
        v[v < 0] = 0.0
        cum = np.cumsum(v)
        xs = np.arange(1, cum.size + 1)
        plt.plot(xs, cum, label=label, color=colors_by_label[label], linewidth=2)

    plt.title(title, size=8)
    plt.xlabel("Request number")
    plt.ylabel("Cumulative processing time (seconds)")
    plt.grid(True, which="both", axis="both", linewidth=0.5, alpha=0.4)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_wer_cdf(
    wer_by_label: Dict[str, np.ndarray],
    colors_by_label: Dict[str, str],
    out_path: str,
    title: str,
    xscale: str = "linear",
):
    plt.figure(figsize=(9, 5))

    for label, wer_values in wer_by_label.items():
        v = np.asarray(wer_values, dtype=np.float64)
        v = v[np.isfinite(v)]

        if v.size == 0:
            continue

        xs = np.sort(v)
        ys = np.arange(1, xs.size + 1) / xs.size

        plt.step(
            xs,
            ys,
            where="post",
            label=label,
            color=colors_by_label[label],
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel("Per-utterance WER")
    plt.ylabel("CDF")
    plt.ylim(0, 1.0)

    if xscale == "log":
        plt.xscale("log")
        plt.xlim(left=1e-4)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to benchmark CSV log (e.g., log.csv).")
    ap.add_argument("--out_dir", default="figs", help="Output directory for plots.")
    ap.add_argument("--prefix", default="", help="Optional filename prefix for output plots.")
    ap.add_argument("--latency_scale", default="log", choices=["linear", "log"], help="Y-axis scale for latency/RTF boxplots.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    prefixes = find_model_prefixes(df)

    label_by_prefix = {p: pretty_label(p) for p in prefixes}
    labels = sorted(label_by_prefix.values(), key=model_sort_key)

    prefix_by_label = {lab: p for p, lab in label_by_prefix.items()}

    colors_by_label = assign_colors(labels)

    wer_by_label: Dict[str, np.ndarray] = {}
    lat_by_label: Dict[str, np.ndarray] = {}
    rtf_by_label: Dict[str, np.ndarray] = {}

    dur = pd.to_numeric(df.get("audio_duration_sec", pd.Series([np.nan] * len(df))), errors="coerce").to_numpy(dtype=np.float64)

    for lab in labels:
        p = prefix_by_label[lab]
        wer = get_series(df, p, "_wer")
        lat = get_series(df, p, "_latency_sec")

        rtf = np.full_like(lat, np.nan, dtype=np.float64)
        valid = np.isfinite(lat) & np.isfinite(dur) & (dur > 0)
        rtf[valid] = lat[valid] / dur[valid]

        wer_by_label[lab] = wer[np.isfinite(wer)]
        lat_by_label[lab] = lat[np.isfinite(lat)]
        rtf_by_label[lab] = rtf[np.isfinite(rtf)]

    n = len(df)
    split = df["split"].iloc[0] if "split" in df.columns and len(df["split"]) else "unknown"

    save_boxplot(
        data_by_label=wer_by_label,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, f"{args.prefix}wer_boxplot.png"),
        title=f"WER by model (split={split}, n={n})",
        ylabel="WER (lower is better)",
        yscale="log",
        y_min=0.0,
    )

    save_boxplot(
        data_by_label=lat_by_label,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, f"{args.prefix}latency_boxplot.png"),
        title=f"Latency by model (split={split}, n={n})",
        ylabel="Latency per request (seconds)",
        yscale=args.latency_scale,
    )

    save_boxplot(
        data_by_label=rtf_by_label,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, f"{args.prefix}rtf_boxplot.png"),
        title=f"RTF by model (split={split}, n={n})",
        ylabel="RTF = latency / audio duration (1.0 = real time)",
        yscale=args.latency_scale,
        hline=1.0,
    )

    save_latency_cdf(
        lat_by_label=lat_by_label,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, f"{args.prefix}latency_cdf.png"),
        title=f"Latency CDF (split={split}, n={n})",
    )

    plot_wer_cdf(
        wer_by_label=wer_by_label,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, "wer_cdf.png"),
        title=f"WER CDF (split={split}, n={n})",
        xscale="log",  # or "log"
    )

    if "sample_index" in df.columns:
        df2 = df.sort_values("sample_index").reset_index(drop=True)
    else:
        df2 = df.reset_index(drop=True)

    lat_by_label_ordered: Dict[str, np.ndarray] = {}
    for lab in labels:
        p = prefix_by_label[lab]
        lat = pd.to_numeric(df2.get(f"{p}_latency_sec", np.nan), errors="coerce").to_numpy(dtype=np.float64)
        lat_by_label_ordered[lab] = lat

    save_cumulative_time(
        lat_by_label=lat_by_label_ordered,
        colors_by_label=colors_by_label,
        out_path=os.path.join(args.out_dir, f"{args.prefix}latency_cumulative_time.png"),
        title=f"Cumulative processing time vs request number (split={split}, n={n})",
    )

    print("Saved:")
    print(os.path.join(args.out_dir, f"{args.prefix}wer_boxplot.png"))
    print(os.path.join(args.out_dir, f"{args.prefix}latency_boxplot.png"))
    print(os.path.join(args.out_dir, f"{args.prefix}rtf_boxplot.png"))
    print(os.path.join(args.out_dir, f"{args.prefix}latency_cdf.png"))
    print(os.path.join(args.out_dir, f"{args.prefix}latency_cumulative_time.png"))


if __name__ == "__main__":
    main()