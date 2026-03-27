import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from ultralytics import YOLO
import torch
from transformers import pipeline

from tqdm import tqdm
from collections import defaultdict

import cv2

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

VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

@dataclass
class Config:
    yolo_weights_path: str = "weights/best.pt"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45

    gemma_model_id: str = "google/gemma-3n-e2b-it"
    gemma_max_new_tokens: int = 120
    gemma_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gemma_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    fire_video_dir: str = "detectium_fire/fire"
    nonfire_video_dir: str = "detectium_fire/non_fire"
    video_max_videos: int = 100

    max_frames_per_video: int = 20
    periodic_k: int = 5

    out_dir: str = "output"
    plots_dir: str = "output/plots"
    tables_dir: str = "output/tables"

CFG = Config()

def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs(cfg.tables_dir, exist_ok=True)


def list_videos(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in VID_EXTS:
            out.append(p)
    return out

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def prf_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "accuracy": float(acc)}


def latency_summary_ms(lat_ms: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(lat_ms, dtype=float).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "min_ms", "max_ms", "n"]}
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "n": int(arr.size),
    }

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_by_scenario(summary: pd.DataFrame, out_path: str) -> None:
    scenarios = ["fire_videos", "nonfire_videos"]
    methods = ["yolo", "gemma_full", "periodic_k5"]

    fig, ax = plt.subplots(figsize=(7, 4))

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
    ax.set_xticklabels(["Fire videos", "No-fire videos"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by scenario (fire vs no-fire)")
    ax.legend()
    save_fig(out_path)


def plot_accuracy_combined(summary: pd.DataFrame, out_path: str) -> None:
    methods = ["yolo", "gemma_full", "periodic_k5"]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    vals = []
    for m in methods:
        row = summary[(summary["scenario"] == "combined") & (summary["method"] == m)]
        vals.append(float(row["accuracy"].iloc[0]) if len(row) else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, edgecolor="black", color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (combined across fire + no-fire)")
    save_fig(out_path)


def plot_latency_cdf(samples: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

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
    fig, ax = plt.subplots(figsize=(6, 4))

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
        "gemma_full": "s",
        "periodic_k5": "D",
    }
    df = summary[summary["scenario"] == "combined"].copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    for _, r in df.iterrows():
        method_key = str(r["method"])

        ax.scatter(
            r["cumulative_latency_s"],
            r["accuracy"],
            marker=METHOD_MARKERS.get(method_key, "o"),
            s=140,
            facecolor=METHOD_COLORS.get(method_key),
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )

        ax.annotate(
            METHOD_LABELS.get(method_key, method_key),
            (r["cumulative_latency_s"], r["accuracy"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_xlabel("Cumulative latency to process all samples (s)")
    ax.set_ylabel("Accuracy (combined)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Tradeoff: cumulative latency vs accuracy")
    save_fig(out_path)

def load_yolo(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def load_gemma_pipe(model_id: str, device: str, dtype):
    if device == "cuda":
        return pipeline("image-text-to-text", model=model_id, device=device, dtype=dtype)
    return pipeline("image-text-to-text", model=model_id, device=device)


VALIDATION_PROMPT = """You are a safety-critical vision assistant.

Task: Decide whether there is REAL fire in the physical scene.
- Bright orange light, sunsets, lamps, reflections, orange objects: NO_REAL_FIRE.
- Flames on a screen/TV/phone only: NO_REAL_FIRE.
- Only answer REAL_FIRE if it is actual burning in the environment.

Return ONLY a JSON object with keys:
  decision: "REAL_FIRE" or "NO_REAL_FIRE"
  reason: short string
"""


def yolo_predict_pil(model: YOLO, pil_img: Image.Image, conf: float, iou: float) -> Tuple[int, float, float]:
    t0 = time.perf_counter()
    results = model.predict(pil_img, conf=conf, iou=iou, verbose=False)
    t1 = time.perf_counter()

    max_conf = 0.0
    r0 = results[0]
    if r0.boxes is not None and len(r0.boxes) > 0:
        confs = r0.boxes.conf.detach().cpu().numpy().tolist()
        max_conf = max(confs) if confs else 0.0

    pred = 1 if max_conf > 0.0 else 0
    return int(pred), float(max_conf), float((t1 - t0) * 1000.0)


def gemma_predict_pil(pipe, pil_img: Image.Image, max_new_tokens: int) -> Tuple[int, str, float]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": VALIDATION_PROMPT}]},
    ]

    t0 = time.perf_counter()
    out = pipe(text=messages, do_sample=False, max_new_tokens=max_new_tokens)
    t1 = time.perf_counter()

    raw = None
    try:
        raw = out[0]["generated_text"][-1]["content"]
    except Exception:
        raw = str(out)

    decision = "NO_REAL_FIRE"
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            cand = json.loads(raw[start:end + 1])
            decision = str(cand.get("decision", "NO_REAL_FIRE")).strip().upper()
    except Exception:
        decision = "NO_REAL_FIRE"

    pred = 1 if decision == "REAL_FIRE" else 0
    return int(pred), decision, float((t1 - t0) * 1000.0)


def sample_frame_indices_uniform(num_frames_total: int, max_samples: int) -> List[int]:
    if num_frames_total <= 0:
        return []
    n = min(int(max_samples), int(num_frames_total))
    if n <= 0:
        return []
    if n == num_frames_total:
        return list(range(num_frames_total))
    idx = np.linspace(0, num_frames_total - 1, n)
    idx = np.round(idx).astype(int)

    out = []
    seen = set()
    for i in idx.tolist():
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def iter_sampled_frames(video_path: str, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_frame_indices_uniform(total, max_frames)

    if not indices:
        cap.release()
        return

    for j, fi in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        yield j, int(fi), pil_img

    cap.release()

def format_prog(stats: dict) -> str:
    s = []
    if "samples" in stats:
        s.append(f"samples={stats['samples']}")
    if "yolo_ms" in stats and stats["samples"]:
        s.append(f"yolo_mean={stats['yolo_ms']/stats['samples']:.1f}ms")
    if "gemma_ms" in stats and stats["samples"]:
        s.append(f"gemma_mean={stats['gemma_ms']/stats['samples']:.1f}ms")
    if "periodic_ms" in stats and stats["samples"]:
        s.append(f"periodic_mean={stats['periodic_ms']/stats['samples']:.1f}ms")
    return " | ".join(s)

def eval_video_folder(
    scenario: str,
    video_paths: List[str],
    y_true_label: int,
    yolo_model: YOLO,
    gemma_pipe,
    cfg: Config,
    global_sample_offset: int = 0,
) -> Tuple[pd.DataFrame, int]:

    rows = []
    global_sample_idx = int(global_sample_offset)

    vids = video_paths[: cfg.video_max_videos]

    stats = defaultdict(float)
    stats["samples"] = 0

    video_pbar = tqdm(
        vids,
        desc=f"[{scenario}] videos",
        unit="video",
        dynamic_ncols=True,
        leave=True,
    )

    for v_i, vp in enumerate(video_pbar, start=1):
        expected_n = None
        try:
            import cv2
            cap = cv2.VideoCapture(vp)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                expected_n = len(sample_frame_indices_uniform(total, cfg.max_frames_per_video))
            else:
                cap.release()
        except Exception:
            expected_n = None

        frame_iter = iter_sampled_frames(vp, cfg.max_frames_per_video)

        frame_pbar = tqdm(
            frame_iter,
            desc=f"  {os.path.basename(vp)}",
            total=expected_n,
            unit="frame",
            dynamic_ncols=True,
            leave=False,
        )

        for sample_in_vid, frame_idx, pil_img in frame_pbar:
            global_sample_idx += 1

            yolo_pred, yolo_conf, lat_yolo_ms = yolo_predict_pil(
                yolo_model, pil_img, cfg.yolo_conf, cfg.yolo_iou
            )

            gem_pred, gem_dec, lat_gemma_full_ms = gemma_predict_pil(
                gemma_pipe, pil_img, cfg.gemma_max_new_tokens
            )

            periodic_called = (global_sample_idx % int(cfg.periodic_k) == 0)
            periodic_pred = int(yolo_pred)
            periodic_dec = None
            lat_periodic_ms = float(lat_yolo_ms)

            if periodic_called:
                gp, gd, lat_g_ms = gemma_predict_pil(gemma_pipe, pil_img, cfg.gemma_max_new_tokens)
                periodic_pred = int(gp)
                periodic_dec = gd
                lat_periodic_ms = float(lat_yolo_ms + lat_g_ms)

            rows.append({
                "scenario": scenario,
                "y_true": int(y_true_label),
                "video_path": vp,
                "video_index": int(v_i),
                "sample_in_video": int(sample_in_vid),
                "frame_index": int(frame_idx),
                "sample_idx_global": int(global_sample_idx),
                "yolo_pred": int(yolo_pred),
                "yolo_conf": float(yolo_conf),
                "lat_yolo_ms": float(lat_yolo_ms),
                "gemma_full_pred": int(gem_pred),
                "gemma_full_decision": str(gem_dec),
                "lat_gemma_full_ms": float(lat_gemma_full_ms),
                "periodic_k": int(cfg.periodic_k),
                "periodic_called": bool(periodic_called),
                "periodic_pred": int(periodic_pred),
                "periodic_gemma_decision": (str(periodic_dec) if periodic_dec is not None else None),
                "lat_periodic_ms": float(lat_periodic_ms),
            })

            stats["samples"] += 1
            stats["yolo_ms"] += float(lat_yolo_ms)
            stats["gemma_ms"] += float(lat_gemma_full_ms)
            stats["periodic_ms"] += float(lat_periodic_ms)

            frame_pbar.set_postfix_str(format_prog(stats))

        video_pbar.set_postfix_str(format_prog(stats))

    df = pd.DataFrame(rows)
    return df, int(global_sample_idx)

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [
        ("yolo", "yolo_pred", "lat_yolo_ms"),
        ("gemma_full", "gemma_full_pred", "lat_gemma_full_ms"),
        ("periodic_k5", "periodic_pred", "lat_periodic_ms"),
    ]

    for scenario in ["fire_videos", "nonfire_videos"]:
        dsc = df[df["scenario"] == scenario]
        if len(dsc) == 0:
            continue

        y_true = dsc["y_true"].to_numpy().astype(int)
        for method, pred_col, lat_col in methods:
            y_pred = dsc[pred_col].to_numpy().astype(int)
            c = confusion_counts(y_true, y_pred)
            prf = prf_from_counts(c)
            lat = latency_summary_ms(dsc[lat_col].to_numpy(dtype=float))
            rows.append({
                "scenario": scenario,
                "method": method,
                "n_samples": int(len(dsc)),
                "accuracy": prf["accuracy"],
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "tp": c["tp"], "tn": c["tn"], "fp": c["fp"], "fn": c["fn"],
                "cumulative_latency_s": float(np.nansum(dsc[lat_col].to_numpy(dtype=float)) / 1000.0),
                **{f"lat_{k}": v for k, v in lat.items()},
            })

    if len(df):
        y_true = df["y_true"].to_numpy().astype(int)
        for method, pred_col, lat_col in methods:
            y_pred = df[pred_col].to_numpy().astype(int)
            c = confusion_counts(y_true, y_pred)
            prf = prf_from_counts(c)
            lat = latency_summary_ms(df[lat_col].to_numpy(dtype=float))
            rows.append({
                "scenario": "combined",
                "method": method,
                "n_samples": int(len(df)),
                "accuracy": prf["accuracy"],
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "tp": c["tp"], "tn": c["tn"], "fp": c["fp"], "fn": c["fn"],
                "cumulative_latency_s": float(np.nansum(df[lat_col].to_numpy(dtype=float)) / 1000.0),
                **{f"lat_{k}": v for k, v in lat.items()},
            })

    return pd.DataFrame(rows)

def main() -> None:
    ensure_dirs(CFG)

    yolo_model = load_yolo(CFG.yolo_weights_path)

    gemma_pipe = load_gemma_pipe(CFG.gemma_model_id, CFG.gemma_device, CFG.gemma_dtype)

    fire_videos = list_videos(CFG.fire_video_dir)
    nonfire_videos = list_videos(CFG.nonfire_video_dir)

    df_fire, next_idx = eval_video_folder(
        "fire_videos", fire_videos, 1, yolo_model, gemma_pipe, CFG, global_sample_offset=0
    )

    df_nonfire, _ = eval_video_folder(
        "nonfire_videos", nonfire_videos, 0, yolo_model, gemma_pipe, CFG, global_sample_offset=next_idx
    )

    df_all = pd.concat([df_fire, df_nonfire], ignore_index=True) if (len(df_fire) or len(df_nonfire)) else pd.DataFrame()

    samples_csv = os.path.join(CFG.tables_dir, "samples.csv")
    df_all.to_csv(samples_csv, index=False)

    summary = compute_summary(df_all)
    summary_csv = os.path.join(CFG.tables_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)

    metrics = {
        "config": asdict(CFG),
        "n_total_samples": int(len(df_all)),
        "summary_rows": json.loads(summary.to_json(orient="records")),
        "display_labels": METHOD_LABELS,
        "palette": PALETTE,
    }
    metrics_path = os.path.join(CFG.tables_dir, "summary_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if len(summary):
        plot_accuracy_by_scenario(summary, os.path.join(CFG.plots_dir, "accuracy_by_scenario.png"))
        plot_accuracy_combined(summary, os.path.join(CFG.plots_dir, "accuracy_combined.png"))
        plot_tradeoff_cumlat_vs_accuracy(summary, os.path.join(CFG.plots_dir, "tradeoff_cumlat_vs_accuracy.png"))

    if len(df_all):
        plot_latency_cdf(df_all, os.path.join(CFG.plots_dir, "latency_cdf.png"))
        plot_latency_boxplot(df_all, os.path.join(CFG.plots_dir, "latency_boxplot.png"))

if __name__ == "__main__":
    main()