import os
import time
import math
import statistics
import warnings
import gc
import random
import zipfile
import json
from contextlib import contextmanager
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RESULTS_DIR = "results_figures_coco"
os.makedirs(RESULTS_DIR, exist_ok=True)

COCO_ROOT = "./coco"
COCO_IMAGES_DIR = os.path.join(COCO_ROOT, "val2017")
COCO_ANN_DIR = os.path.join(COCO_ROOT, "annotations")
COCO_ANN_FILE = os.path.join(COCO_ANN_DIR, "instances_val2017.json")

COCO_VAL2017_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"

EXPERIMENT_ROUNDS = 1
RESAMPLE_EACH_ROUND = False
RANDOM_SEED = 42

WAVE_WINDOW = 100

NUM_IMAGES = 10000
ONLY_IMAGES_WITH_PERSON = True

TARGET_NAME = "person"
COCO_PERSON_CATEGORY_ID = 1

NUM_RUNS = 5
WARMUP_RUNS = 1
CONF_THRESH = 0.7

FAST_MODE = False
if FAST_MODE:
    NUM_RUNS = 2
    WARMUP_RUNS = 0
    NUM_IMAGES = min(NUM_IMAGES, 20)
    EXPERIMENT_ROUNDS = 1

DEVICE_PREF = "cuda" if torch.cuda.is_available() else "cpu"

BLIP2_MODEL_ID = "Salesforce/blip2-flan-t5-xl"
BLIP2_PROMPT = "How many people are visible in this image? Answer with a single integer."
BLIP2_MAX_NEW_TOKENS = 20

GEMMA3N_E2B_ID = "google/gemma-3n-e2b-it"
GEMMA3N_E4B_ID = "google/gemma-3n-e4b-it"
GEMMA3N_QUESTION = "How many people are visible in this image? Answer with a single integer."
GEMMA3N_MAX_NEW_TOKENS = 20

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@contextmanager
def model_scope(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    cleanup_cuda()

def parse_first_int(text):
    import re
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else 0

def update_pr(pred, gt, stats):
    stats["tp"] += min(pred, gt)
    stats["fp"] += max(0, pred - gt)
    stats["fn"] += max(0, gt - pred)

def precision_recall(stats):
    p = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-9)
    r = stats["tp"] / (stats["tp"] + stats["fn"] + 1e-9)
    return float(p), float(r)

def summarize_metrics(preds, gts, times):
    preds = np.array(preds, dtype=np.float32)
    gts = np.array(gts, dtype=np.float32)
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(math.sqrt(np.mean((preds - gts) ** 2)))
    mean_t = float(np.mean(times)) if len(times) else 0.0
    std_t = float(np.std(times)) if len(times) else 0.0
    return mae, rmse, mean_t, std_t

def latency_percentiles(times_s, ps=(50, 90, 95, 99)):
    arr = np.asarray(times_s, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in ps}
    vals = np.percentile(arr, list(ps))
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, name))
    plt.close(fig)

def is_zip_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig == b"PK\x03\x04"
    except Exception:
        return False

def robust_download(url: str, dest_path: str, retries: int = 3, timeout: int = 60):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading ({attempt}/{retries}): {url}", flush=True)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            with open(dest_path, "wb") as f:
                f.write(data)

            if is_zip_file(dest_path):
                print(f"Saved ZIP: {dest_path} ({len(data)/1e6:.1f} MB)", flush=True)
                return

            snippet = data[:200]
            snippet_txt = snippet.decode("utf-8", errors="replace")
            print(f"ERROR: downloaded file is not a ZIP: {dest_path}", flush=True)
            print(f"First 200 bytes:\n{snippet_txt}\n", flush=True)

        except Exception as e:
            print(f"Unexpected download error: {e}", flush=True)

        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass

        time.sleep(2)

    raise RuntimeError(f"Failed to download a valid ZIP from {url}. Destination was {dest_path}.")

def extract_zip(zip_path: str, dest_dir: str):
    print(f"Extracting: {zip_path} -> {dest_dir}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

def ensure_coco_val2017():
    images_ok = os.path.isdir(COCO_IMAGES_DIR) and len(os.listdir(COCO_IMAGES_DIR)) > 0
    anns_ok = os.path.isfile(COCO_ANN_FILE)
    os.makedirs(COCO_ROOT, exist_ok=True)

    if not images_ok:
        val_zip = os.path.join(COCO_ROOT, "val2017.zip")
        if os.path.exists(val_zip) and not is_zip_file(val_zip):
            print(f"Found corrupted/non-zip file, removing: {val_zip}", flush=True)
            os.remove(val_zip)
        if not os.path.exists(val_zip):
            robust_download(COCO_VAL2017_URL, val_zip)
        extract_zip(val_zip, COCO_ROOT)

    if not anns_ok:
        ann_zip = os.path.join(COCO_ROOT, "annotations_trainval2017.zip")
        if os.path.exists(ann_zip) and not is_zip_file(ann_zip):
            print(f"Found corrupted/non-zip file, removing: {ann_zip}", flush=True)
            os.remove(ann_zip)
        if not os.path.exists(ann_zip):
            robust_download(COCO_ANN_URL, ann_zip)
        extract_zip(ann_zip, COCO_ROOT)

def torchvision_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    from torchvision.transforms import functional as F
    return F.to_tensor(img_pil)

def write_csv(path: str, header: list[str], rows: list[list]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            out = []
            for x in r:
                s = str(x)
                if "," in s or "\n" in s or '"' in s:
                    s = '"' + s.replace('"', '""') + '"'
                out.append(s)
            f.write(",".join(out) + "\n")

def dump_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def sample_coco_images(seed: int):
    ensure_coco_val2017()

    from torchvision.datasets import CocoDetection
    coco_ds = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANN_FILE)

    idxs = list(range(len(coco_ds)))
    random.seed(seed)
    random.shuffle(idxs)

    selected = []
    for idx in idxs:
        img_pil, anns = coco_ds[idx]
        gt_person = sum(1 for a in anns if int(a.get("category_id", -1)) == COCO_PERSON_CATEGORY_ID)
        if ONLY_IMAGES_WITH_PERSON and gt_person == 0:
            continue

        img_id = coco_ds.ids[idx]
        file_name = coco_ds.coco.loadImgs([img_id])[0]["file_name"]
        path = os.path.join(COCO_IMAGES_DIR, file_name)

        img_cv = cv2.imread(path)
        img_tensor = torchvision_to_tensor(img_pil)

        selected.append((file_name, path, gt_person, img_cv, img_pil, img_tensor))
        if len(selected) >= NUM_IMAGES:
            break

    gts = [x[2] for x in selected]
    return selected, gts


def run_yolo(samples):
    from ultralytics import YOLO
    preds = []
    times_all = []
    lat_runs = []
    lat_per_image = []
    pr = {"tp": 0, "fp": 0, "fn": 0}
    runs, vars_ = [], []

    with model_scope("YOLOv8"):
        model = YOLO("yolov8n.pt")

        for i, (fname, path, gt, img_cv, img_pil, img_tensor) in enumerate(samples):
            print(f"[YOLO] {i+1}/{len(samples)} {fname}", flush=True)
            per_preds = []
            per_lats = []

            for r in range(NUM_RUNS):
                t0 = time.perf_counter()
                res = model(img_cv, verbose=False)
                sync_if_cuda()
                t1 = time.perf_counter()

                c = sum(int(cls == 0) for rr in res for cls in rr.boxes.cls)
                if r >= WARMUP_RUNS:
                    per_preds.append(int(c))
                    lat = (t1 - t0)
                    per_lats.append(lat)
                    times_all.append(lat)

            pred = int(statistics.median(per_preds)) if per_preds else 0
            preds.append(pred)
            update_pr(pred, gt, pr)
            runs.append(per_preds)
            vars_.append(float(np.std(per_preds)) if per_preds else 0.0)

            lat_runs.append(per_lats)
            lat_per_image.append(float(statistics.median(per_lats)) if per_lats else 0.0)

        del model

    return preds, times_all, lat_per_image, lat_runs, pr, runs, vars_


def run_detr(samples):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    preds, times_all = [], []
    lat_runs, lat_per_image = [], []
    pr = {"tp": 0, "fp": 0, "fn": 0}
    runs, vars_ = [], []
    device = DEVICE_PREF

    with model_scope("DETR"):
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval().to(device)

        for i, (fname, path, gt, img_cv, img_pil, img_tensor) in enumerate(samples):
            print(f"[DETR] {i+1}/{len(samples)} {fname}", flush=True)

            per_preds, per_lats = [], []
            target_sizes = torch.tensor([img_pil.size[::-1]], device=device)

            for r in range(NUM_RUNS):
                inp = processor(images=img_pil, return_tensors="pt")
                inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}

                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(**inp)
                post = processor.post_process_object_detection(
                    out, threshold=CONF_THRESH, target_sizes=target_sizes
                )[0]
                sync_if_cuda()
                t1 = time.perf_counter()

                c = int((post["labels"] == 1).sum().item())

                del inp, out, post
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if r >= WARMUP_RUNS:
                    per_preds.append(c)
                    lat = (t1 - t0)
                    per_lats.append(lat)
                    times_all.append(lat)

            pred = int(statistics.median(per_preds)) if per_preds else 0
            preds.append(pred)
            update_pr(pred, gt, pr)
            runs.append(per_preds)
            vars_.append(float(np.std(per_preds)) if per_preds else 0.0)
            lat_runs.append(per_lats)
            lat_per_image.append(float(statistics.median(per_lats)) if per_lats else 0.0)

        del model, processor

    return preds, times_all, lat_per_image, lat_runs, pr, runs, vars_


def run_frcnn(samples):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    preds = []
    times_all = []
    lat_runs = []
    lat_per_image = []
    pr = {"tp": 0, "fp": 0, "fn": 0}
    runs, vars_ = [], []
    device = DEVICE_PREF

    with model_scope("FRCNN"):
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(device)

        for i, (fname, path, gt, img_cv, img_pil, img_tensor) in enumerate(samples):
            print(f"[FRCNN] {i+1}/{len(samples)} {fname}", flush=True)
            per_preds = []
            per_lats = []
            it = img_tensor.to(device)

            for r in range(NUM_RUNS):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model([it])[0]
                sync_if_cuda()
                t1 = time.perf_counter()

                c = sum((int(l) == 1) and (float(s) > CONF_THRESH)
                        for l, s in zip(out["labels"], out["scores"]))
                if r >= WARMUP_RUNS:
                    per_preds.append(int(c))
                    lat = (t1 - t0)
                    per_lats.append(lat)
                    times_all.append(lat)

            pred = int(statistics.median(per_preds)) if per_preds else 0
            preds.append(pred)
            update_pr(pred, gt, pr)
            runs.append(per_preds)
            vars_.append(float(np.std(per_preds)) if per_preds else 0.0)

            lat_runs.append(per_lats)
            lat_per_image.append(float(statistics.median(per_lats)) if per_lats else 0.0)

        del model

    return preds, times_all, lat_per_image, lat_runs, pr, runs, vars_


def run_blip2(samples):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    preds = []
    times_all = []
    lat_runs = []
    lat_per_image = []
    pr = {"tp": 0, "fp": 0, "fn": 0}
    runs, vars_ = [], []
    device = DEVICE_PREF
    dtype = torch.float16 if device == "cuda" else torch.float32

    with model_scope("BLIP2"):
        processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
        model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_ID, torch_dtype=dtype
        ).to(device).eval()

        for i, (fname, path, gt, img_cv, img_pil, img_tensor) in enumerate(samples):
            print(f"[BLIP2] {i+1}/{len(samples)} {fname}", flush=True)
            per_preds = []
            per_lats = []

            for r in range(NUM_RUNS):
                t0 = time.perf_counter()
                inp = processor(images=img_pil, text=BLIP2_PROMPT, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=BLIP2_MAX_NEW_TOKENS)
                txt = processor.decode(gen[0], skip_special_tokens=True)
                c = parse_first_int(txt)
                sync_if_cuda()
                t1 = time.perf_counter()

                if r >= WARMUP_RUNS:
                    per_preds.append(int(c))
                    lat = (t1 - t0)
                    per_lats.append(lat)
                    times_all.append(lat)

            pred = int(statistics.median(per_preds)) if per_preds else 0
            preds.append(pred)
            update_pr(pred, gt, pr)
            runs.append(per_preds)
            vars_.append(float(np.std(per_preds)) if per_preds else 0.0)

            lat_runs.append(per_lats)
            lat_per_image.append(float(statistics.median(per_lats)) if per_lats else 0.0)

        del model, processor

    return preds, times_all, lat_per_image, lat_runs, pr, runs, vars_


def run_gemma3n(samples, model_id: str, tag: str):
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    preds = []
    times_all = []
    lat_runs = []
    lat_per_image = []
    pr = {"tp": 0, "fp": 0, "fn": 0}
    runs, vars_ = [], []
    device = DEVICE_PREF
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with model_scope(tag):
        processor = AutoProcessor.from_pretrained(model_id)
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device).eval()

        for i, (fname, path, gt, img_cv, img_pil, img_tensor) in enumerate(samples):
            print(f"[{tag}] {i+1}/{len(samples)} {fname}", flush=True)
            per_preds = []
            per_lats = []

            for r in range(NUM_RUNS):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_pil},
                            {"type": "text", "text": GEMMA3N_QUESTION},
                        ],
                    },
                ]

                t0 = time.perf_counter()
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                for k, v in inputs.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        inputs[k] = v.to(dtype=dtype)

                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=GEMMA3N_MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                gen_tokens = generation[0][input_len:]
                txt = processor.decode(gen_tokens, skip_special_tokens=True)
                c = parse_first_int(txt)

                sync_if_cuda()
                t1 = time.perf_counter()

                if r >= WARMUP_RUNS:
                    per_preds.append(int(c))
                    lat = (t1 - t0)
                    per_lats.append(lat)
                    times_all.append(lat)

            pred = int(statistics.median(per_preds)) if per_preds else 0
            preds.append(pred)
            update_pr(pred, gt, pr)
            runs.append(per_preds)
            vars_.append(float(np.std(per_preds)) if per_preds else 0.0)

            lat_runs.append(per_lats)
            lat_per_image.append(float(statistics.median(per_lats)) if per_lats else 0.0)

        del model, processor

    return preds, times_all, lat_per_image, lat_runs, pr, runs, vars_

def compute_model_metrics(name: str, preds, gts, times, pr):
    mae, rmse, mean_t, std_t = summarize_metrics(preds, gts, times)
    p, r = precision_recall(pr)
    pct = latency_percentiles(times, ps=(50, 90, 95, 99))
    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Precision": p,
        "Recall": r,
        "Latency_mean_s": mean_t,
        "Latency_std_s": std_t,
        "Latency_mean_ms": mean_t * 1000.0,
        "Latency_std_ms": std_t * 1000.0,
        "Latency_p50_s": pct["p50"],
        "Latency_p90_s": pct["p90"],
        "Latency_p95_s": pct["p95"],
        "Latency_p99_s": pct["p99"],
        "Latency_p50_ms": pct["p50"] * 1000.0,
        "Latency_p90_ms": pct["p90"] * 1000.0,
        "Latency_p95_ms": pct["p95"] * 1000.0,
        "Latency_p99_ms": pct["p99"] * 1000.0,
    }

def aggregate_across_rounds(round_metrics: list[dict]):
    by_model = {}
    for rm in round_metrics:
        for model_name, md in rm["metrics_by_model"].items():
            by_model.setdefault(model_name, []).append(md)

    out = {}
    for model_name, lst in by_model.items():
        keys = [k for k in lst[0].keys() if k != "model"]
        agg = {}
        for k in keys:
            vals = [float(x[k]) for x in lst]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        out[model_name] = agg
    return out

def rolling_percentiles(series, window: int = 200, ps=(50, 90, 95, 99)):
    x = np.asarray(series, dtype=np.float64)
    n = x.shape[0]
    out = {p: np.empty(n, dtype=np.float64) for p in ps}

    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = x[lo:i+1]
        vals = np.percentile(chunk, list(ps))
        for p, v in zip(ps, vals):
            out[p][i] = v

    return out

def plot_latency_wave(latency_ms, model_name: str, out_name: str, window: int = 200):
    ps = (50, 90, 95, 99)
    roll = rolling_percentiles(latency_ms, window=window, ps=ps)

    x = np.arange(len(latency_ms), dtype=np.int32)
    raw = np.asarray(latency_ms, dtype=np.float64)

    fig = plt.figure(figsize=(12, 3.5))
    ax = plt.gca()

    green = "#2ca02c"
    purple = "#9467bd"
    blue = "#1f77b4"
    red = "#d62728"

    ax.fill_between(x, 0, roll[50], color=green, alpha=0.25, label="≤ p50")

    ax.fill_between(x, roll[50], roll[90], color=purple, alpha=0.25, label="p50–p90")

    ax.fill_between(x, roll[90], roll[95], color=blue, alpha=0.30, label="p90–p95")

    ax.fill_between(x, roll[95], roll[99], color=red, alpha=0.35, label="p95–p99")

    ax.plot(x, roll[50], color=green, linewidth=1.2)
    ax.plot(x, roll[90], color=purple, linewidth=1.2)
    ax.plot(x, roll[95], color=blue, linewidth=1.2)
    ax.plot(x, roll[99], color=red, linewidth=1.2)

    ax.plot(x, raw, color="black", linewidth=0.6, alpha=0.15, label="raw latency")

    ax.set_title(f"Latency percentile wave — {model_name} (window={window})")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    save_fig(fig, out_name)

def plot_latency_percentiles_last_round(metrics_by_model, out_name="latency_percentiles.png"):
    """
    Visualize latency percentiles per model (p50, p90, p95, p99) using the
    metrics_by_model dict from the last round.
    """
    order = ["YOLO", "DETR", "FRCNN", "BLIP2", "GEMMA_E2B", "GEMMA_E4B"]
    labels = order

    p50 = [metrics_by_model[m]["Latency_p50_ms"] for m in order]
    p90 = [metrics_by_model[m]["Latency_p90_ms"] for m in order]
    p95 = [metrics_by_model[m]["Latency_p95_ms"] for m in order]
    p99 = [metrics_by_model[m]["Latency_p99_ms"] for m in order]

    xs = np.arange(len(order), dtype=np.float32)

    fig = plt.figure()
    ax = plt.gca()

    ax.vlines(xs, p50, p99)

    ax.scatter(xs, p50, marker="o", label="p50")
    ax.scatter(xs, p90, marker="^", label="p90")
    ax.scatter(xs, p95, marker="s", label="p95")
    ax.scatter(xs, p99, marker="x", label="p99")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency percentiles per model (last round)")
    ax.legend()

    save_fig(fig, out_name)

def plot_last_round(per_image, gts, times_map, maes_map):
    models = ["YOLO", "DETR", "FRCNN", "BLIP2", "GEMMA_E2B", "GEMMA_E4B"]
    keys = ["yolo", "detr", "frcnn", "blip2", "gemma_e2b", "gemma_e4b"]

    fig = plt.figure()
    x = range(len(gts))
    plt.plot(x, per_image["gt"], label="GT", linewidth=2, color="black")
    for k in keys:
        plt.plot(x, per_image[k], label=k.upper())
    plt.xlabel("Image index")
    plt.ylabel(f"{TARGET_NAME} count")
    plt.legend()
    plt.title("Predicted count per image (COCO val2017)")
    save_fig(fig, "line_pred_vs_gt.png")

    fig = plt.figure()
    for k in keys:
        plt.plot([abs(p - g) for p, g in zip(per_image[k], gts)], label=k.upper())
    plt.xlabel("Image index")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.title("Per-image absolute error")
    save_fig(fig, "abs_error_per_image.png")

    fig = plt.figure()
    correct = [sum(p == g for p, g in zip(per_image[k], gts)) for k in keys]
    incorrect = [len(gts) - c for c in correct]
    plt.bar(models, correct, label="Correct")
    plt.bar(models, incorrect, bottom=correct, label="Incorrect")
    plt.ylabel("Number of images")
    plt.legend()
    plt.title("Correct vs incorrect predictions")
    save_fig(fig, "correct_vs_incorrect.png")

    fig = plt.figure()
    for k in ["yolo", "blip2", "gemma_e2b", "gemma_e4b"]:
        plt.hist([p - g for p, g in zip(per_image[k], gts)], alpha=0.6, label=k.upper())
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Signed error (pred - gt)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Overcount vs undercount behavior")
    save_fig(fig, "signed_error_hist.png")

    fig = plt.figure()
    xs = [times_map[m] for m in models]
    ys = [maes_map[m] for m in models]
    plt.scatter(xs, ys)
    for x0, y0, m in zip(xs, ys, models):
        plt.text(x0, y0, m)
    plt.xlabel("Latency mean (s)")
    plt.ylabel("MAE")
    plt.title("Accuracy–latency tradeoff")
    save_fig(fig, "tradeoff_mae_latency.png")

    fig = plt.figure()
    var_lists = [
        per_image["yolo_var"], per_image["detr_var"], per_image["frcnn_var"],
        per_image["blip2_var"], per_image["gemma_e2b_var"], per_image["gemma_e4b_var"]
    ]
    plt.boxplot(var_lists, labels=models)
    plt.ylabel("Std. dev. of predicted count")
    plt.title("Prediction stability across runs")
    save_fig(fig, "prediction_variability.png")

    fig = plt.figure()
    plt.scatter(gts, per_image["blip2_var"], label="BLIP2")
    plt.scatter(gts, per_image["gemma_e2b_var"], label="GEMMA_E2B")
    plt.scatter(gts, per_image["gemma_e4b_var"], label="GEMMA_E4B")
    plt.scatter(gts, per_image["yolo_var"], label="YOLO")
    plt.xlabel("Ground truth count")
    plt.ylabel("Prediction std. dev.")
    plt.legend()
    plt.title("Prediction instability vs crowd size")
    save_fig(fig, "instability_vs_gt.png")


def main():
    config = {
        "DEVICE_PREF": DEVICE_PREF,
        "NUM_IMAGES": NUM_IMAGES,
        "ONLY_IMAGES_WITH_PERSON": ONLY_IMAGES_WITH_PERSON,
        "TARGET_NAME": TARGET_NAME,
        "COCO_PERSON_CATEGORY_ID": COCO_PERSON_CATEGORY_ID,
        "NUM_RUNS": NUM_RUNS,
        "WARMUP_RUNS": WARMUP_RUNS,
        "CONF_THRESH": CONF_THRESH,
        "EXPERIMENT_ROUNDS": EXPERIMENT_ROUNDS,
        "RESAMPLE_EACH_ROUND": RESAMPLE_EACH_ROUND,
        "RANDOM_SEED": RANDOM_SEED,
        "BLIP2_MODEL_ID": BLIP2_MODEL_ID,
        "GEMMA3N_E2B_ID": GEMMA3N_E2B_ID,
        "GEMMA3N_E4B_ID": GEMMA3N_E4B_ID,
    }
    dump_json(os.path.join(RESULTS_DIR, "config.json"), config)

    print(f"Device preference: {DEVICE_PREF}", flush=True)
    print(f"Task: COCO val2017 '{TARGET_NAME}' instance counting (GT category_id={COCO_PERSON_CATEGORY_ID})", flush=True)
    print(f"Rounds: {EXPERIMENT_ROUNDS} | Resample each round: {RESAMPLE_EACH_ROUND}", flush=True)

    fixed_samples = None
    fixed_gts = None
    if not RESAMPLE_EACH_ROUND:
        fixed_samples, fixed_gts = sample_coco_images(RANDOM_SEED)
        print(f"Fixed sample selected: {len(fixed_samples)} images", flush=True)

    all_rounds_summary_rows = []
    all_rounds_metrics = []
    last_round_artifacts = None

    for round_idx in range(1, EXPERIMENT_ROUNDS + 1):

        if RESAMPLE_EACH_ROUND:
            samples, gts = sample_coco_images(RANDOM_SEED + round_idx)
        else:
            samples, gts = fixed_samples, fixed_gts

        yolo_preds, yolo_times, yolo_lat_img, yolo_lat_runs, yolo_pr, yolo_runs, yolo_vars = run_yolo(samples)
        detr_preds, detr_times, detr_lat_img, detr_lat_runs, detr_pr, detr_runs, detr_vars = run_detr(samples)
        frcnn_preds, frcnn_times, frcnn_lat_img, frcnn_lat_runs, frcnn_pr, frcnn_runs, frcnn_vars = run_frcnn(samples)
        blip2_preds, blip2_times, blip2_lat_img, blip2_lat_runs, blip2_pr, blip2_runs, blip2_vars = run_blip2(samples)
        gemma_e2b_preds, gemma_e2b_times, gemma_e2b_lat_img, gemma_e2b_lat_runs, gemma_e2b_pr, gemma_e2b_runs, gemma_e2b_vars = run_gemma3n(
            samples, GEMMA3N_E2B_ID, "Gemma3n-E2B"
        )
        gemma_e4b_preds, gemma_e4b_times, gemma_e4b_lat_img, gemma_e4b_lat_runs, gemma_e4b_pr, gemma_e4b_runs, gemma_e4b_vars = run_gemma3n(
            samples, GEMMA3N_E4B_ID, "Gemma3n-E4B"
        )

        per_image = {
            "gt": gts,
            "yolo": yolo_preds,
            "detr": detr_preds,
            "frcnn": frcnn_preds,
            "blip2": blip2_preds,
            "gemma_e2b": gemma_e2b_preds,
            "gemma_e4b": gemma_e4b_preds,
            "yolo_var": yolo_vars,
            "detr_var": detr_vars,
            "frcnn_var": frcnn_vars,
            "blip2_var": blip2_vars,
            "gemma_e2b_var": gemma_e2b_vars,
            "gemma_e4b_var": gemma_e4b_vars,
            "yolo_lat_ms": [x * 1000.0 for x in yolo_lat_img],
            "detr_lat_ms": [x * 1000.0 for x in detr_lat_img],
            "frcnn_lat_ms": [x * 1000.0 for x in frcnn_lat_img],
            "blip2_lat_ms": [x * 1000.0 for x in blip2_lat_img],
            "gemma_e2b_lat_ms": [x * 1000.0 for x in gemma_e2b_lat_img],
            "gemma_e4b_lat_ms": [x * 1000.0 for x in gemma_e4b_lat_img],
        }

        metrics_by_model = {
            "YOLO": compute_model_metrics("YOLO", yolo_preds, gts, yolo_times, yolo_pr),
            "DETR": compute_model_metrics("DETR", detr_preds, gts, detr_times, detr_pr),
            "FRCNN": compute_model_metrics("FRCNN", frcnn_preds, gts, frcnn_times, frcnn_pr),
            "BLIP2": compute_model_metrics("BLIP2", blip2_preds, gts, blip2_times, blip2_pr),
            "GEMMA_E2B": compute_model_metrics("GEMMA_E2B", gemma_e2b_preds, gts, gemma_e2b_times, gemma_e2b_pr),
            "GEMMA_E4B": compute_model_metrics("GEMMA_E4B", gemma_e4b_preds, gts, gemma_e4b_times, gemma_e4b_pr),
        }

        for k, md in metrics_by_model.items():
            print(
                f"{k:<10} | MAE={md['MAE']:.3f} RMSE={md['RMSE']:.3f} "
                f"P={md['Precision']:.3f} R={md['Recall']:.3f} "
                f"T(mean)={md['Latency_mean_ms']:.1f}±{md['Latency_std_ms']:.1f}ms "
                f"p95={md['Latency_p95_ms']:.1f}ms",
                flush=True
            )

        per_image_csv = os.path.join(RESULTS_DIR, f"round_{round_idx}_per_image.csv")
        header = [
            "file_name", "gt",
            "yolo", "detr", "frcnn", "blip2", "gemma_e2b", "gemma_e4b",
            "yolo_var", "detr_var", "frcnn_var", "blip2_var", "gemma_e2b_var", "gemma_e4b_var",
            "yolo_lat_ms", "detr_lat_ms", "frcnn_lat_ms", "blip2_lat_ms", "gemma_e2b_lat_ms", "gemma_e4b_lat_ms",
        ]
        rows = []
        for i, (fname, path, gt, *_rest) in enumerate(samples):
            rows.append([
                fname, gt,
                yolo_preds[i], detr_preds[i], frcnn_preds[i], blip2_preds[i], gemma_e2b_preds[i], gemma_e4b_preds[i],
                yolo_vars[i], detr_vars[i], frcnn_vars[i], blip2_vars[i], gemma_e2b_vars[i], gemma_e4b_vars[i],
                yolo_lat_img[i] * 1000.0, detr_lat_img[i] * 1000.0, frcnn_lat_img[i] * 1000.0,
                blip2_lat_img[i] * 1000.0, gemma_e2b_lat_img[i] * 1000.0, gemma_e4b_lat_img[i] * 1000.0,
            ])
        write_csv(per_image_csv, header, rows)

        round_metrics_obj = {
            "round": round_idx,
            "num_images": len(samples),
            "metrics_by_model": metrics_by_model,

        }
        dump_json(os.path.join(RESULTS_DIR, f"round_{round_idx}_metrics.json"), round_metrics_obj)

        for model_name, md in metrics_by_model.items():
            all_rounds_summary_rows.append([
                round_idx, model_name,
                md["MAE"], md["RMSE"], md["Precision"], md["Recall"],
                md["Latency_mean_ms"], md["Latency_std_ms"],
                md["Latency_p50_ms"], md["Latency_p90_ms"], md["Latency_p95_ms"], md["Latency_p99_ms"],
            ])

        all_rounds_metrics.append(round_metrics_obj)

        last_round_artifacts = (per_image, gts, metrics_by_model)

    all_rounds_csv = os.path.join(RESULTS_DIR, "all_rounds_metrics.csv")
    write_csv(
        all_rounds_csv,
        [
            "round", "model",
            "MAE", "RMSE", "Precision", "Recall",
            "Latency_mean_ms", "Latency_std_ms",
            "Latency_p50_ms", "Latency_p90_ms", "Latency_p95_ms", "Latency_p99_ms",
        ],
        all_rounds_summary_rows
    )

    agg = aggregate_across_rounds(all_rounds_metrics)
    dump_json(os.path.join(RESULTS_DIR, "final_aggregate_metrics.json"), agg)

    for model_name, stats_map in agg.items():
        mae_mu = stats_map["MAE"]["mean"]
        mae_sd = stats_map["MAE"]["std"]
        rmse_mu = stats_map["RMSE"]["mean"]
        rmse_sd = stats_map["RMSE"]["std"]
        lat_mu = stats_map["Latency_mean_ms"]["mean"]
        lat_sd = stats_map["Latency_mean_ms"]["std"]
        p95_mu = stats_map["Latency_p95_ms"]["mean"]
        p95_sd = stats_map["Latency_p95_ms"]["std"]
        p_mu = stats_map["Precision"]["mean"]
        p_sd = stats_map["Precision"]["std"]
        r_mu = stats_map["Recall"]["mean"]
        r_sd = stats_map["Recall"]["std"]
        print(
            f"{model_name:<10} | MAE={mae_mu:.3f}±{mae_sd:.3f} "
            f"RMSE={rmse_mu:.3f}±{rmse_sd:.3f} "
            f"P={p_mu:.3f}±{p_sd:.3f} R={r_mu:.3f}±{r_sd:.3f} "
            f"Lat(mean ms)={lat_mu:.1f}±{lat_sd:.1f} "
            f"Lat(p95 ms)={p95_mu:.1f}±{p95_sd:.1f}",
            flush=True
        )

    if last_round_artifacts is not None:
        per_image, gts, metrics_by_model = last_round_artifacts
        times_map = {
            "YOLO": metrics_by_model["YOLO"]["Latency_mean_s"],
            "DETR": metrics_by_model["DETR"]["Latency_mean_s"],
            "FRCNN": metrics_by_model["FRCNN"]["Latency_mean_s"],
            "BLIP2": metrics_by_model["BLIP2"]["Latency_mean_s"],
            "GEMMA_E2B": metrics_by_model["GEMMA_E2B"]["Latency_mean_s"],
            "GEMMA_E4B": metrics_by_model["GEMMA_E4B"]["Latency_mean_s"],
        }
        maes_map = {
            "YOLO": metrics_by_model["YOLO"]["MAE"],
            "DETR": metrics_by_model["DETR"]["MAE"],
            "FRCNN": metrics_by_model["FRCNN"]["MAE"],
            "BLIP2": metrics_by_model["BLIP2"]["MAE"],
            "GEMMA_E2B": metrics_by_model["GEMMA_E2B"]["MAE"],
            "GEMMA_E4B": metrics_by_model["GEMMA_E4B"]["MAE"],
        }
        plot_last_round(per_image, gts, times_map, maes_map)
        plot_latency_percentiles_last_round(metrics_by_model, out_name="latency_percentiles.png")

        plot_latency_wave(per_image["yolo_lat_ms"], "YOLO", "latency_wave_yolo.png", window=WAVE_WINDOW)
        plot_latency_wave(per_image["detr_lat_ms"], "DETR", "latency_wave_detr.png", window=WAVE_WINDOW)
        plot_latency_wave(per_image["frcnn_lat_ms"], "FRCNN", "latency_wave_frcnn.png", window=WAVE_WINDOW)
        plot_latency_wave(per_image["blip2_lat_ms"], "BLIP2", "latency_wave_blip2.png", window=WAVE_WINDOW)
        plot_latency_wave(per_image["gemma_e2b_lat_ms"], "GEMMA_E2B", "latency_wave_gemma_e2b.png", window=WAVE_WINDOW)
        plot_latency_wave(per_image["gemma_e4b_lat_ms"], "GEMMA_E4B", "latency_wave_gemma_e4b.png", window=WAVE_WINDOW)

if __name__ == "__main__":
    main()