import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torchvision

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def choose_label_from_text(text: str, label_names: List[str]) -> Tuple[str, Optional[str]]:
    t = norm(text)
    if not label_names:
        return "", "no_labels"

    n2o = {norm(x): x for x in label_names}
    if t in n2o:
        return n2o[t], None

    hits = []
    for ln in label_names:
        nln = norm(ln)
        if nln and nln in t:
            hits.append((len(nln), ln))
    if hits:
        hits.sort(reverse=True)
        return hits[0][1], None

    tset = set(t.split())
    best = (0.0, label_names[0])
    for ln in label_names:
        lset = set(norm(ln).split())
        if not lset:
            continue
        score = len(tset & lset) / max(1, len(lset))
        if score > best[0]:
            best = (score, ln)

    if best[0] == 0.0:
        return best[1], "mapping_failed"
    return best[1], None


def load_hf_subset(dataset: str, config: Optional[str], split: str, num_samples: int, seed: int):
    from datasets import load_dataset

    def _sample(ds):
        n = min(num_samples, len(ds))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(ds), size=n, replace=False).tolist()
        return ds.select(idxs)

    try:
        ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
        return _sample(ds)
    except RuntimeError as e:
        parquet_glob = f"hf://datasets/{dataset}@refs/convert/parquet/{config}/{split}/*.parquet"
        ds = load_dataset("parquet", data_files=parquet_glob, split="train")
        return _sample(ds)


def infer_video_and_label_columns(ds) -> Tuple[str, str, List[str]]:
    from datasets import ClassLabel

    video_col = None
    label_col = None
    label_names: List[str] = []

    for k, feat in ds.features.items():
        if feat.__class__.__name__.lower() == "video":
            video_col = k
            break
    if video_col is None:
        for k in ("video", "clip", "vid", "path"):
            if k in ds.column_names:
                video_col = k
                break

    for k, feat in ds.features.items():
        if isinstance(feat, ClassLabel):
            label_col = k
            label_names = list(feat.names)
            break
    if label_col is None:
        for k in ("label", "labels", "action", "category", "class"):
            if k in ds.column_names:
                label_col = k
                break

    if not label_names:
        vals = ds[label_col]
        if vals and isinstance(vals[0], str):
            label_names = sorted(set(vals))
        else:
            uniq = sorted(set(int(v) for v in vals))
            label_names = [str(u) for u in uniq]

    return video_col, label_col, label_names

def _is_torchcodec_decoder(x) -> bool:
    return hasattr(x, "metadata") and (
        hasattr(x, "get_frames_at") or hasattr(x, "get_frames_in_range") or hasattr(x, "__getitem__")
    )


def _frames_uint8_nhwc_from_torch_tensor(t) -> np.ndarray:
    import torch

    if isinstance(t, torch.Tensor):
        x = t
    else:
        x = getattr(t, "data", None)
        if x is None:
            raise RuntimeError("Unknown torchcodec frame container (no .data)")

    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.cpu().numpy().astype(np.uint8, copy=False)


def sample_frames_for_models(
    video_item,
    clip_len: int,
    stride: int,
    gemma_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if _is_torchcodec_decoder(video_item):
        dec = video_item
        num_frames = int(getattr(dec.metadata, "num_frames", 0) or 0)
        if num_frames <= 0:
            num_frames = len(dec)

        needed = 1 + (clip_len - 1) * stride
        if num_frames >= needed:
            start = (num_frames - needed) // 2
            tv_idxs = [start + i * stride for i in range(clip_len)]
        else:
            tv_idxs = [(i * stride) % num_frames for i in range(clip_len)]

        k = max(1, int(gemma_frames))
        if num_frames > 1:
            gem_idxs = np.linspace(0, num_frames - 1, num=k).astype(int).tolist()
        else:
            gem_idxs = [0] * k

        all_idxs = sorted(set(tv_idxs + gem_idxs))
        batch = dec.get_frames_at(indices=all_idxs)
        all_frames = _frames_uint8_nhwc_from_torch_tensor(batch)

        idx_to_pos = {idx: j for j, idx in enumerate(all_idxs)}
        tv = np.stack([all_frames[idx_to_pos[ii]] for ii in tv_idxs], axis=0)
        gm = np.stack([all_frames[idx_to_pos[ii]] for ii in gem_idxs], axis=0)
        return tv, gm

    path = None
    tmp_cleanup = None
    if isinstance(video_item, str):
        path = video_item
    elif isinstance(video_item, dict):
        p = video_item.get("path")
        b = video_item.get("bytes", None)
        if isinstance(p, str) and os.path.exists(p):
            path = p
        elif b is not None:
            import tempfile

            tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tf.write(b)
            tf.flush()
            tf.close()
            path = tf.name
            tmp_cleanup = tf.name
        elif isinstance(p, str):
            path = p

    from torchvision.io import read_video

    video, _audio, _info = read_video(path, pts_unit="sec")
    if tmp_cleanup and os.path.exists(tmp_cleanup):
        os.remove(tmp_cleanup)

    if video.shape[-1] != 3:
        video = video[..., :3]

    frames = video.cpu().numpy().astype(np.uint8, copy=False)
    T = frames.shape[0]

    needed = 1 + (clip_len - 1) * stride
    if T >= needed:
        start = (T - needed) // 2
        tv_idxs = [start + i * stride for i in range(clip_len)]
    else:
        tv_idxs = [(i * stride) % T for i in range(clip_len)]
    tv = frames[tv_idxs]

    k = max(1, int(gemma_frames))
    if T > 1:
        gem_idxs = np.linspace(0, T - 1, num=k).astype(int)
    else:
        gem_idxs = np.zeros((k,), dtype=int)
    gm = frames[gem_idxs]

    return tv, gm

def preprocess_kinetics_clip(clip: np.ndarray, out_size: int = 112) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
    T, C, H, W = x.shape

    short = min(H, W)
    if short != 128:
        scale = 128.0 / float(short)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    _, _, H2, W2 = x.shape
    top = max(0, (H2 - out_size) // 2)
    left = max(0, (W2 - out_size) // 2)
    x = x[:, :, top : top + out_size, left : left + out_size]

    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return x

@dataclass
class TorchvisionActionRunner:
    model_name: str
    device: str
    allowed_labels: List[str]

    _model = None
    _categories: Optional[List[str]] = None

    def _ensure_model(self):
        if self._model is not None:
            return

        

        ctor = getattr(torchvision.models.video, self.model_name)

        weights_enum = None
        for a in dir(torchvision.models.video):
            if a.endswith("_Weights"):
                enum_obj = getattr(torchvision.models.video, a)
                if hasattr(enum_obj, "DEFAULT") and self.model_name.replace("_", "") in a.lower().replace("_", ""):
                    weights_enum = enum_obj
                    break
        if weights_enum is None:
            for a in dir(torchvision.models.video):
                if a.endswith("_Weights"):
                    enum_obj = getattr(torchvision.models.video, a)
                    if hasattr(enum_obj, "DEFAULT"):
                        try:
                            _ = ctor(weights=enum_obj.DEFAULT)
                            weights_enum = enum_obj
                            break
                        except Exception:
                            continue

        weights = weights_enum.DEFAULT
        self._model = ctor(weights=weights).eval().to(self.device)

        cats = None
        try:
            cats = weights.meta.get("categories", None)
        except Exception:
            cats = None
        self._categories = cats if isinstance(cats, list) else None

    def predict_label(self, clip_tensor: "torch.Tensor") -> Tuple[str, str, Optional[str]]:
        self._ensure_model()
        import torch

        with torch.inference_mode():
            logits = self._model(clip_tensor.to(self.device))
            pred_idx = int(torch.argmax(logits, dim=1).item())

        raw = self._categories[pred_idx] if self._categories and 0 <= pred_idx < len(self._categories) else str(pred_idx)
        pred, note = choose_label_from_text(raw, self.allowed_labels)
        return pred, raw, note

def make_frame_grid(frames: np.ndarray, cols: int = 4, tile_size: int = 224) -> "PIL.Image.Image":
    from PIL import Image

    N = frames.shape[0]
    cols = max(1, int(cols))
    rows = int(np.ceil(N / cols))
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), (0, 0, 0))

    for i in range(N):
        r = i // cols
        c = i % cols
        img = Image.fromarray(frames[i])
        img = img.resize((tile_size, tile_size))
        canvas.paste(img, (c * tile_size, r * tile_size))
    return canvas


@dataclass
class Gemma3nActionRunner:
    model_id: str
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 8

    _processor = None
    _model = None

    def _ensure_model(self):
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        torch_dtype = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        ).eval()

    def predict_label(self, grid_image, allowed_labels: List[str], prompt: str) -> Tuple[str, str, Optional[str]]:
        self._ensure_model()
        import torch

        allowed = ", ".join(allowed_labels)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a careful action recognition assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": grid_image},
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n\n"
                            f"Allowed labels: {allowed}\n"
                            "Return EXACTLY one label from the allowed labels. No extra words."
                        ),
                    },
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._model.dtype)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            out = out[0][input_len:]

        raw = self._processor.decode(out, skip_special_tokens=True).strip()
        pred, note = choose_label_from_text(raw, allowed_labels)
        return pred, raw, note

def plot_latency_boxplot(latencies_by_model: Dict[str, List[float]], path: str, title: str, yscale: str):
    import matplotlib.pyplot as plt

    labels, data = [], []
    for name, vals in latencies_by_model.items():
        v = [float(x) for x in vals if x is not None and np.isfinite(x)]
        labels.append(f"{name}\n(med={np.median(v):.3f}s)" if v else f"{name}\n(med=n/a)")
        data.append(v)

    plt.figure(figsize=(max(10, 2 + 2.2 * len(labels)), 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Latency per sample (seconds)")
    plt.yscale(yscale)

    if yscale == "log":
        all_vals = [x for v in data for x in v if x > 0]
        if all_vals:
            plt.ylim(bottom=max(min(all_vals) * 0.8, 1e-4))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_bar(acc_by_model: Dict[str, float], path: str, title: str):
    import matplotlib.pyplot as plt

    names = list(acc_by_model.keys())
    vals = [acc_by_model[n] for n in names]

    plt.figure(figsize=(max(10, 2 + 1.2 * len(names)), 5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Top-1 Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="nateraw/kinetics-mini")
    ap.add_argument("--config", default="", help="Parquet fallback")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tv_models", default="r3d_18")

    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--crop_size", type=int, default=112)

    ap.add_argument("--use_gemma_e2b", action="store_true")
    ap.add_argument("--use_gemma_e4b", action="store_true")
    ap.add_argument("--gemma_device_map", default="auto")
    ap.add_argument("--gemma_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gemma_max_new_tokens", type=int, default=8)
    ap.add_argument("--gemma_prompt", default="Recognize the action shown in the video frames")
    ap.add_argument("--gemma_frames", type=int, default=8)
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--tile_size", type=int, default=224)

    ap.add_argument("--latency_scale", default="log", choices=["linear", "log"])
    ap.add_argument("--latency_plot_path", default="ar_latency_kinetics.png")
    ap.add_argument("--acc_plot_path", default="ar_accuracy_kinetics.png")
    ap.add_argument("--log_path", default="ar_kinetics_log.csv")
    ap.add_argument("--save_jsonl", action="store_true")
    ap.add_argument("--jsonl_path", default="ar_kinetics_log.jsonl")

    args = ap.parse_args()
    config = args.config.strip() or None

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    print(f"\nLoading dataset={args.dataset} config={config} split={args.split} n={args.num_samples} seed={args.seed} ...")
    ds = load_hf_subset(args.dataset, config, args.split, args.num_samples, args.seed)

    video_col, label_col, label_names = infer_video_and_label_columns(ds)
    print(f"Using video_col={video_col}, label_col={label_col}, num_labels={len(label_names)}")
    if len(label_names) <= 50:
        print("Labels:", label_names)

    runners: Dict[str, object] = {}
    tv_models = [m.strip() for m in args.tv_models.split(",") if m.strip()]
    for m in tv_models:
        runners[f"TV:{m}"] = TorchvisionActionRunner(model_name=m, device=device, allowed_labels=label_names)

    if args.use_gemma_e2b:
        runners["Gemma3n_e2b_it"] = Gemma3nActionRunner(
            model_id="google/gemma-3n-e2b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )
    if args.use_gemma_e4b:
        runners["Gemma3n_e4b_it"] = Gemma3nActionRunner(
            model_id="google/gemma-3n-e4b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )

    model_names = list(runners.keys())

    latencies: Dict[str, List[float]] = {mn: [] for mn in model_names}
    corrects: Dict[str, List[int]] = {mn: [] for mn in model_names}
    records: List[Dict] = []

    for i in range(len(ds)):
        ex = ds[i]

        true_val = ex[label_col]
        if isinstance(true_val, str):
            true_label = true_val
        else:
            try:
                true_label = label_names[int(true_val)]
            except Exception:
                true_label = str(true_val)

        row = {
            "sample_index": i + 1,
            "dataset": args.dataset,
            "config": config or "",
            "split": args.split,
            "true_label": true_label,
        }

        try:
            tv_clip, gemma_fr = sample_frames_for_models(
                ex[video_col],
                clip_len=args.clip_len,
                stride=args.stride,
                gemma_frames=args.gemma_frames,
            )
        except Exception as e:
            for mn in model_names:
                col = mn.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_")
                row[f"{col}_pred"] = ""
                row[f"{col}_raw"] = ""
                row[f"{col}_latency_sec"] = 0.0
                row[f"{col}_correct"] = 0
                row[f"{col}_note"] = f"decode_error:{repr(e)}"
                latencies[mn].append(0.0)
                corrects[mn].append(0)
            records.append(row)
            continue

        # prepare inputs once
        try:
            clip_tensor = preprocess_kinetics_clip(tv_clip, out_size=args.crop_size)
            clip_err = None
        except Exception as e:
            clip_tensor = None
            clip_err = repr(e)

        try:
            grid_img = make_frame_grid(gemma_fr, cols=args.grid_cols, tile_size=args.tile_size)
            grid_err = None
        except Exception as e:
            grid_img = None
            grid_err = repr(e)

        for mn, runner in runners.items():
            t0 = time.perf_counter()
            pred, raw, note = "", "", None
            try:
                if isinstance(runner, TorchvisionActionRunner):
                    if clip_tensor is None:
                        raise RuntimeError(f"clip_prep_failed:{clip_err}")
                    pred, raw, note = runner.predict_label(clip_tensor)
                else:
                    if grid_img is None:
                        raise RuntimeError(f"grid_prep_failed:{grid_err}")
                    pred, raw, note = runner.predict_label(grid_img, label_names, args.gemma_prompt)
            except Exception as e:
                pred, raw, note = "", "", repr(e)

            dt = time.perf_counter() - t0
            is_correct = int(norm(pred) == norm(true_label))

            latencies[mn].append(dt)
            corrects[mn].append(is_correct)

            col = mn.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_")
            row[f"{col}_pred"] = pred
            row[f"{col}_raw"] = raw
            row[f"{col}_latency_sec"] = dt
            row[f"{col}_correct"] = is_correct
            if note:
                row[f"{col}_note"] = note

        records.append(row)

        if (i + 1) % 5 == 0 or (i + 1) == len(ds):
            print(f"  processed {i+1}/{len(ds)}")

    acc_by_model: Dict[str, float] = {}
    for mn in model_names:
        lat = np.array(latencies[mn], dtype=np.float64)
        acc = float(np.mean(corrects[mn])) if corrects[mn] else 0.0
        acc_by_model[mn] = acc

        p50 = float(np.percentile(lat, 50))
        p90 = float(np.percentile(lat, 90))
        mean = float(lat.mean())

        print(f"\nModel: {mn}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Latency (sec): mean={mean:.3f}  p50={p50:.3f}  p90={p90:.3f}")

    plot_latency_boxplot(
        latencies,
        args.latency_plot_path,
        f"Action latency ({args.dataset}, split={args.split}, n={len(records)})",
        yscale=args.latency_scale,
    )
    print(f"\nSaved latency plot to: {args.latency_plot_path}")

    plot_accuracy_bar(
        acc_by_model,
        args.acc_plot_path,
        f"Action accuracy ({args.dataset}, split={args.split}, n={len(records)})",
    )
    print(f"Saved accuracy plot to: {args.acc_plot_path}")

    pd.DataFrame(records).to_csv(args.log_path, index=False)
    print(f"Saved per-sample CSV log to: {args.log_path}")

    if args.save_jsonl:
        with open(args.jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved per-sample JSONL log to: {args.jsonl_path}")

if __name__ == "__main__":
    main()
