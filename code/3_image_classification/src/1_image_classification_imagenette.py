import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torchvision

import numpy as np
import pandas as pd

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

def load_image_subset(dataset_name: str, config: Optional[str], split: str, num_samples: int, seed: int):
    from datasets import load_dataset

    def _sample(ds):
        n = min(num_samples, len(ds))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(ds), size=n, replace=False).tolist()
        return ds.select(idxs)

    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        return _sample(ds)

    except RuntimeError as e:
        msg = str(e)
        print(msg)

        parquet_glob = f"hf://datasets/{dataset_name}@refs/convert/parquet/{config}/{split}/*.parquet"
        ds = load_dataset("parquet", data_files=parquet_glob, split="train")
        return _sample(ds)

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def infer_image_and_label_columns(ds) -> Tuple[str, str, List[str]]:
    from datasets import ClassLabel

    image_col = None
    label_col = None
    label_names: List[str] = []

    for k, feat in ds.features.items():
        if feat.__class__.__name__ == "Image":
            image_col = k
            break
    if image_col is None:
        for k in ("image", "img"):
            if k in ds.column_names:
                image_col = k
                break

    for k, feat in ds.features.items():
        if isinstance(feat, ClassLabel):
            label_col = k
            label_names = list(feat.names)
            break

    if label_col is None:
        for k in ("label", "labels", "class", "category"):
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

    return image_col, label_col, label_names

@dataclass
class TorchvisionImagenetSubsetRunner:
    model_name: str
    device: str
    subset_labels: List[str]

    _model = None
    _weights = None
    _preprocess = None
    _categories: Optional[List[str]] = None
    _subset_indices: Optional[List[int]] = None

    def _ensure_model(self):
        if self._model is not None:
            return

        ctor = getattr(torchvision.models, self.model_name)

        weights_enum = None
        for a in dir(torchvision.models):
            if a.endswith("_Weights"):
                enum_obj = getattr(torchvision.models, a)
                if hasattr(enum_obj, "DEFAULT"):
                    if self.model_name.replace("_", "") in a.lower().replace("_", ""):
                        weights_enum = enum_obj
                        break
        if weights_enum is None:
            for a in dir(torchvision.models):
                if a.endswith("_Weights"):
                    enum_obj = getattr(torchvision.models, a)
                    if hasattr(enum_obj, "DEFAULT"):
                        try:
                            _ = ctor(weights=enum_obj.DEFAULT)
                            weights_enum = enum_obj
                            break
                        except Exception:
                            continue

        weights = weights_enum.DEFAULT
        model = ctor(weights=weights).eval().to(self.device)
        preprocess = weights.transforms()

        cats = None
        try:
            cats = weights.meta.get("categories", None)
        except Exception:
            cats = None
        if not cats or not isinstance(cats, list):
            raise RuntimeError("Could not find ImageNet categories in weights.meta['categories'].")

        self._weights = weights
        self._model = model
        self._preprocess = preprocess
        self._categories = cats

        subset_indices: List[int] = []
        cat_norm = [norm(c) for c in cats]
        for lab in self.subset_labels:
            nlab = norm(lab)
            hits = [i for i, nc in enumerate(cat_norm) if nc == nlab]
            if not hits:
                hits = [i for i, nc in enumerate(cat_norm) if (nlab in nc) or (nc in nlab)]
            if not hits:
                continue
            subset_indices.extend(hits)

        subset_indices = sorted(set(subset_indices))

        self._subset_indices = subset_indices

    def predict_label(self, pil_image) -> Tuple[str, Optional[str]]:
        self._ensure_model()
        import torch

        x = self._preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self._model(x)[0]  # [1000]

        idxs = self._subset_indices or []
        if not idxs:
            return "", "no_subset_indices"

        sub_logits = logits[idxs]
        best_pos = int(torch.argmax(sub_logits).item())
        imagenet_idx = idxs[best_pos]
        imagenet_cat = self._categories[imagenet_idx] if self._categories else str(imagenet_idx)

        pred_label, note = choose_label_from_text(imagenet_cat, self.subset_labels)
        return pred_label, note

@dataclass
class Gemma3nVisionLabelRunner:
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

    def predict_label(self, pil_image, subset_labels: List[str], prompt: str) -> Tuple[str, str, Optional[str]]:
        self._ensure_model()
        import torch

        allowed = ", ".join(subset_labels)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a careful image classification assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
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
        pred, note = choose_label_from_text(raw, subset_labels)
        return pred, raw, note


@dataclass
class Qwen3VLVisionLabelRunner:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 8

    _processor = None
    _model = None

    def _ensure_model(self):
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception as e:
            raise RuntimeError(f"Error: {repr(e)}")

        torch_dtype = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        ).eval()

    def predict_label(self, pil_image, subset_labels: List[str], prompt: str) -> Tuple[str, str, Optional[str]]:
        self._ensure_model()
        import torch

        allowed = ", ".join(subset_labels)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a careful image classification assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
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
        pred, note = choose_label_from_text(raw, subset_labels)
        return pred, raw, note

@dataclass
class Ministral3VisionLabelRunner:
    model_id: str = "mistralai/Ministral-3-3B-Base-2512"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 8

    _tokenizer = None
    _model = None
    _device = None

    def _ensure_model(self):
        if self._tokenizer is not None and self._model is not None:
            return

        import torch
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        torch_dtype = {
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]

        self._tokenizer = MistralCommonBackend.from_pretrained(self.model_id)
        self._model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        ).eval()

        try:
            self._device = next(self._model.parameters()).device
        except Exception:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_label(self, pil_image, subset_labels: List[str], prompt: str) -> Tuple[str, str, Optional[str]]:
        self._ensure_model()
        import torch
        import tempfile
        from pathlib import Path

        allowed = ", ".join(subset_labels)

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = Path(f.name)
            pil_image.convert("RGB").save(tmp_path, format="JPEG", quality=95)
            image_url = tmp_path.resolve().as_uri()  # file:///...

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}\n\n"
                                f"Allowed labels: {allowed}\n"
                                "Return EXACTLY one label from the allowed labels. No extra words."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

            tokenized = self._tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

            if "input_ids" in tokenized:
                tokenized["input_ids"] = tokenized["input_ids"].to(device=self._device)
            if "attention_mask" in tokenized:
                tokenized["attention_mask"] = tokenized["attention_mask"].to(device=self._device)

            image_sizes = None
            if "pixel_values" in tokenized:
                tokenized["pixel_values"] = tokenized["pixel_values"].to(device=self._device, dtype=self._model.dtype)
                image_sizes = [tokenized["pixel_values"].shape[-2:]]

            input_len = int(tokenized["input_ids"].shape[-1])

            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            if image_sizes is not None:
                gen_kwargs["image_sizes"] = image_sizes

            with torch.inference_mode():
                out = self._model.generate(**tokenized, **gen_kwargs)[0]
                out = out[input_len:]

            raw = self._tokenizer.decode(out, skip_special_tokens=True).strip()
            pred, note = choose_label_from_text(raw, subset_labels)
            return pred, raw, note

        except Exception as e:
            return "", "", f"Ministral3 failed: {repr(e)}"

        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

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

    ap.add_argument("--dataset", default="frgfm/imagenette",
                    help="HF dataset name (recommended: frgfm/imagenette or frgfm/imagewoof).")
    ap.add_argument("--config", default="160px",
                    help="Dataset config (e.g., 160px, 320px, full). Use empty string to omit.")
    ap.add_argument("--split", default="validation", help="Dataset split (e.g., train/validation).")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument(
        "--tv_models",
        default="mobilenet_v3_large",
        help="Comma-separated torchvision ImageNet-pretrained models (e.g., mobilenet_v3_large,resnet50,convnext_tiny,vit_b_16).",
    )

    ap.add_argument("--use_gemma_e2b", action="store_true")
    ap.add_argument("--use_gemma_e4b", action="store_true")
    ap.add_argument("--gemma_device_map", default="auto")
    ap.add_argument("--gemma_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gemma_max_new_tokens", type=int, default=8)

    ap.add_argument("--use_qwen3_vl_2b", action="store_true")
    ap.add_argument("--qwen3_vl_model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen3_vl_device_map", default="auto")
    ap.add_argument("--qwen3_vl_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--qwen3_vl_max_new_tokens", type=int, default=8)

    ap.add_argument("--use_ministral3_3b", action="store_true")
    ap.add_argument("--ministral3_model_id", default="mistralai/Ministral-3-3B-Base-2512")
    ap.add_argument("--ministral3_device_map", default="auto")
    ap.add_argument("--ministral3_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--ministral3_max_new_tokens", type=int, default=8)

    ap.add_argument("--vlm_prompt", default="Classify the image.",
                    help="Instruction for VLMs (script adds allowed labels + strict output rule).")

    ap.add_argument("--latency_scale", default="log", choices=["linear", "log"])
    ap.add_argument("--latency_plot_path", default="latency_plot.png")
    ap.add_argument("--acc_plot_path", default="acc_plot.png")

    ap.add_argument("--log_path", default="imgcls_benchmark_log.csv")
    ap.add_argument("--save_jsonl", action="store_true")
    ap.add_argument("--jsonl_path", default="imgcls_benchmark_log.jsonl")

    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    config = args.config.strip() or None

    ds = load_image_subset(args.dataset, config, args.split, args.num_samples, args.seed)
    image_col, label_col, label_names = infer_image_and_label_columns(ds)
    n = len(ds)

    runners: Dict[str, object] = {}

    tv_models = [m.strip() for m in args.tv_models.split(",") if m.strip()]

    for m in tv_models:
        runners[f"TV:{m}"] = TorchvisionImagenetSubsetRunner(
            model_name=m,
            device=device,
            subset_labels=label_names,
        )

    if args.use_gemma_e2b:
        runners["Gemma3n:e2b-it"] = Gemma3nVisionLabelRunner(
            model_id="google/gemma-3n-e2b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )
    if args.use_gemma_e4b:
        runners["Gemma3n:e4b-it"] = Gemma3nVisionLabelRunner(
            model_id="google/gemma-3n-e4b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )

    if args.use_qwen3_vl_2b:
        runners["Qwen3-VL:2B-Instruct"] = Qwen3VLVisionLabelRunner(
            model_id=args.qwen3_vl_model_id,
            device_map=args.qwen3_vl_device_map,
            dtype=args.qwen3_vl_dtype,
            max_new_tokens=args.qwen3_vl_max_new_tokens,
        )

    if args.use_ministral3_3b:
        runners["Ministral3:3B-Base-BF16"] = Ministral3VisionLabelRunner(
            model_id=args.ministral3_model_id,
            device_map=args.ministral3_device_map,
            dtype=args.ministral3_dtype,
            max_new_tokens=args.ministral3_max_new_tokens,
        )

    model_names = list(runners.keys())

    latencies: Dict[str, List[float]] = {mn: [] for mn in model_names}
    corrects: Dict[str, List[int]] = {mn: [] for mn in model_names}
    records: List[Dict] = []

    for i, ex in enumerate(ds, 1):
        pil_image = ex[image_col]
        true_label = ex[label_col]
        if not isinstance(true_label, str):
            true_label = label_names[int(true_label)]

        row = {
            "sample_index": i,
            "dataset": args.dataset,
            "config": config or "",
            "split": args.split,
            "true_label": true_label,
        }

        for mn, runner in runners.items():
            t0 = time.perf_counter()
            err = None
            raw = ""
            pred_label = ""

            try:
                if isinstance(runner, TorchvisionImagenetSubsetRunner):
                    pred_label, err = runner.predict_label(pil_image)
                    raw = pred_label
                else:
                    pred_label, raw, err = runner.predict_label(pil_image, label_names, args.vlm_prompt)
            except Exception as e:
                pred_label = ""
                raw = ""
                err = repr(e)

            dt = time.perf_counter() - t0
            is_correct = int(norm(pred_label) == norm(true_label))

            latencies[mn].append(dt)
            corrects[mn].append(is_correct)

            col_prefix = (
                mn.replace(":", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
                .replace(" ", "_")
            )

            row[f"{col_prefix}_pred"] = pred_label
            row[f"{col_prefix}_raw"] = raw
            row[f"{col_prefix}_latency_sec"] = dt
            row[f"{col_prefix}_correct"] = is_correct
            if err:
                row[f"{col_prefix}_note"] = err

        records.append(row)

        if i % 25 == 0 or i == n:
            print(f"  processed {i}/{n}")
            
    acc_by_model: Dict[str, float] = {}
    for mn in model_names:
        lat = np.array(latencies[mn], dtype=np.float64)
        acc = float(np.mean(corrects[mn])) if corrects[mn] else 0.0
        acc_by_model[mn] = acc

        p50 = float(np.percentile(lat, 50))
        p90 = float(np.percentile(lat, 90))
        mean = float(lat.mean())

        print(f"\nModel: {mn}")
        print(f"  Top-1 Accuracy: {acc:.4f}")
        print(f"  Latency (sec): mean={mean:.3f}  p50={p50:.3f}  p90={p90:.3f}")

    lat_title = f"Latency (dataset={args.dataset}, config={config}, split={args.split}, n={n})"
    plot_latency_boxplot(latencies, args.latency_plot_path, lat_title, yscale=args.latency_scale)
    print(f"\nSaved latency plot to: {args.latency_plot_path}")

    acc_title = f"Top-1 Accuracy (dataset={args.dataset}, config={config}, split={args.split}, n={n})"
    plot_accuracy_bar(acc_by_model, args.acc_plot_path, acc_title)
    print(f"Saved accuracy plot to: {args.acc_plot_path}")

    df = pd.DataFrame(records)
    df.to_csv(args.log_path, index=False)

    if args.save_jsonl:
        with open(args.jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
