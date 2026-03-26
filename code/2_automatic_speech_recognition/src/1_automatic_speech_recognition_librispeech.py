import argparse
import json
import os
import re
import time
import tempfile
import shutil
import zipfile
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import transformers
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

import matplotlib.pyplot as plt

import pandas as pd

def normalize_for_wer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def write_wav_pcm16(path: str, audio: np.ndarray, sr: int) -> None:
    audio = audio.astype(np.float32, copy=False)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())

def load_librispeech_subset(split: str, num_samples: int, seed: int):
    from datasets import load_dataset, Audio

    ds = load_dataset("librispeech_asr", split=split)
    n = min(num_samples, len(ds))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False).tolist()

    ds = ds.select(idxs)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

@dataclass
class WhisperRunner:
    model_name: str
    device: str
    compute_type: str
    language: Optional[str] = "en"
    vad_filter: bool = False

    _model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        self._ensure_model()
        segments, _info = self._model.transcribe(
            audio_array,
            language=self.language,
            vad_filter=self.vad_filter,
        )
        return "".join(seg.text for seg in segments).strip()

@dataclass
class Gemma3nAsrRunner:
    model_id: str
    device_map: str
    dtype: str
    max_new_tokens: int = 128

    _processor = None
    _model = None

    def _ensure_model(self):
        if self._processor is not None and self._model is not None:
            return

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

    def transcribe(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        self._ensure_model()
        import torch

        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            write_wav_pcm16(wav_path, audio_array, sampling_rate)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a careful speech transcription assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": wav_path},
                        {"type": "text", "text": "Transcribe the audio verbatim. Output only the transcript."},
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

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                generation = generation[0][input_len:]

            text = self._processor.decode(generation, skip_special_tokens=True).strip()
            return text

VOSK_MODEL_URLS: Dict[str, Tuple[str, str]] = {
    "small-en-us-0.15": (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "vosk-model-small-en-us-0.15",
    ),
    "en-us-0.22": (
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "vosk-model-en-us-0.22",
    ),
}


def _download_file(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "asr-bench/1.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None

        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = (downloaded / total) * 100.0
                print(f"    downloading... {pct:5.1f}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="\r")
        if total:
            print(" " * 80, end="\r")


def ensure_vosk_model(model_key_or_path: str, vosk_root: str) -> str:
    if os.path.isdir(model_key_or_path):
        return os.path.abspath(model_key_or_path)

    key = model_key_or_path.strip()
    if key not in VOSK_MODEL_URLS:
        raise ValueError(
            f"Unknown Vosk model '{key}'. Either pass a local directory path, "
            f"or one of: {', '.join(VOSK_MODEL_URLS.keys())}"
        )

    url, extracted_dir = VOSK_MODEL_URLS[key]
    target_dir = os.path.abspath(os.path.join(vosk_root, extracted_dir))

    if os.path.isdir(target_dir) and os.path.isdir(os.path.join(target_dir, "conf")):
        print(f"Vosk model already present: {target_dir}")
        return target_dir

    os.makedirs(vosk_root, exist_ok=True)
    zip_path = os.path.join(vosk_root, extracted_dir + ".zip")

    print(f"Downloading Vosk model '{key}' from: {url}")
    _download_file(url, zip_path)

    print(f"Extracting to: {vosk_root}")
    tmp_extract = os.path.join(vosk_root, f".tmp_extract_{extracted_dir}")
    if os.path.isdir(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_extract)

    candidate = os.path.join(tmp_extract, extracted_dir)
    if not os.path.isdir(candidate):
        dirs = [d for d in os.listdir(tmp_extract) if os.path.isdir(os.path.join(tmp_extract, d))]
        if len(dirs) == 1:
            candidate = os.path.join(tmp_extract, dirs[0])
        else:
            raise RuntimeError(f"Unexpected zip structure in {zip_path}; extracted dirs: {dirs}")

    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(candidate, target_dir)
    shutil.rmtree(tmp_extract, ignore_errors=True)

    print(f"Vosk model ready: {target_dir}")
    return target_dir

@dataclass
class VoskRunner:
    model_dir: str
    _model = None

    def _ensure_model(self):
        if self._model is None:
            from vosk import Model
            self._model = Model(self.model_dir)

    def transcribe(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        self._ensure_model()
        from vosk import KaldiRecognizer

        if sampling_rate != 16000:
            raise ValueError(f"Vosk expects 16kHz audio. Got {sampling_rate} Hz.")

        audio = np.clip(audio_array.astype(np.float32, copy=False), -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)

        rec = KaldiRecognizer(self._model, 16000)
        rec.SetWords(False)
        rec.AcceptWaveform(audio_i16.tobytes())
        final = rec.FinalResult()

        try:
            j = json.loads(final)
            return (j.get("text") or "").strip()
        except Exception:
            return ""

def compute_wer(refs: List[str], hyps: List[str]) -> float:
    from jiwer import wer
    return float(wer(refs, hyps))


def compute_per_sample_wer(refs: List[str], hyps: List[str]) -> List[float]:
    from jiwer import wer
    return [float(wer([r], [h])) for r, h in zip(refs, hyps)]


def plot_and_save_wer(
    per_sample_by_model: Dict[str, List[float]],
    overall_by_model: Dict[str, float],
    plot_path: str,
    title: str,
):
    labels, data = [], []
    for name, per_sample in per_sample_by_model.items():
        labels.append(f"{name}\n(mean={overall_by_model[name]:.3f})")
        data.append(per_sample)

    plt.figure(figsize=(max(10, 2 + 2.2 * len(labels)), 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("WER (lower is better)")

    ymax = 1.0
    for v in data:
        if v:
            ymax = max(ymax, max(v) * 1.05)
    plt.ylim(0, ymax)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def plot_and_save_latency(
    latencies_by_model: Dict[str, List[float]],
    plot_path: str,
    title: str,
    yscale: str = "log",
):
    

    labels, data = [], []
    for name, vals in latencies_by_model.items():
        v = [float(x) for x in vals if x is not None and np.isfinite(x)]
        labels.append(f"{name}\n(median={np.median(v):.3f}s)" if len(v) else f"{name}\n(median=n/a)")
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
    plt.savefig(plot_path, dpi=200)
    plt.close()


def plot_and_save_rtf(
    rtf_by_model: Dict[str, List[float]],
    plot_path: str,
    title: str,
    yscale: str = "log",
):
    labels, data = [], []
    for name, vals in rtf_by_model.items():
        v = [float(x) for x in vals if x is not None and np.isfinite(x) and x > 0]
        labels.append(f"{name}\n(median={np.median(v):.3f})" if len(v) else f"{name}\n(median=n/a)")
        data.append(v)

    plt.figure(figsize=(max(10, 2 + 2.2 * len(labels)), 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("RTF = latency / audio_duration (lower is better)")
    plt.yscale(yscale)

    plt.axhline(1.0, linestyle="--", linewidth=1)

    if yscale == "log":
        all_vals = [x for v in data for x in v if x > 0]
        if all_vals:
            plt.ylim(bottom=max(min(all_vals) * 0.8, 1e-4))

    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--split",
        default="test.clean",
        choices=["test.clean", "test.other", "dev.clean", "dev.other", "train.clean.100"],
        help="LibriSpeech split (we still only download a subset via --num_samples).",
    )
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--whisper_models",
        default="tiny,base,small,medium,large-v3,large-v3-turbo,large-turbo",
        help=(
            "Comma-separated faster-whisper model names/paths "
            "(e.g., tiny,base,small,medium,large-v3,large-v3-turbo,large-turbo)."
        ),
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute_type", default="auto")
    ap.add_argument("--whisper_vad", action="store_true", help="Enable VAD filter in Whisper (default off).")

    ap.add_argument("--use_gemma_e2b", action="store_true", help="Enable google/gemma-3n-e2b-it (audio->text).")
    ap.add_argument("--use_gemma_e4b", action="store_true", help="Enable google/gemma-3n-e4b-it (audio->text).")
    ap.add_argument("--gemma_device_map", default="auto", help="Transformers device_map (auto/cuda/cpu).")
    ap.add_argument("--gemma_dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gemma_max_new_tokens", type=int, default=128)

    ap.add_argument("--use_vosk", action="store_true", help="Enable Vosk edge ASR.")
    ap.add_argument(
        "--vosk_models",
        default="small-en-us-0.15,en-us-0.22",
    )
    ap.add_argument("--vosk_root", default="./vosk_models", help="Where to download/extract Vosk models (if keys used).")

    ap.add_argument("--plot_path", default="wer_plot.png")
    ap.add_argument("--latency_plot_path", default="latency_plot.png", help="Where to save the latency plot PNG.")
    ap.add_argument("--rtf_plot_path", default="", help="Optional: where to save an RTF plot PNG (empty = skip).")
    ap.add_argument(
        "--latency_scale",
        default="log",
        choices=["linear", "log"],
        help="Latency plot y-axis scale (log is nicer when models differ a lot).",
    )
    ap.add_argument("--log_path", default="results_asr_log_2620.csv")
    ap.add_argument("--save_jsonl", action="store_true")
    ap.add_argument("--jsonl_path", default="results_asr_log_2620.jsonl")

    args = ap.parse_args()

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    whisper_model_names = [m.strip() for m in args.whisper_models.split(",") if m.strip()]

    runners: Dict[str, object] = {}

    for wm in whisper_model_names:
        runners[f"Whisper:{wm}"] = WhisperRunner(
            model_name=wm,
            device=device,
            compute_type=args.compute_type,
            language="en",
            vad_filter=bool(args.whisper_vad),
        )

    if args.use_gemma_e2b:
        runners["Gemma3n:E2B-it"] = Gemma3nAsrRunner(
            model_id="google/gemma-3n-e2b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )

    if args.use_gemma_e4b:
        runners["Gemma3n:E4B-it"] = Gemma3nAsrRunner(
            model_id="google/gemma-3n-e4b-it",
            device_map=args.gemma_device_map,
            dtype=args.gemma_dtype,
            max_new_tokens=args.gemma_max_new_tokens,
        )

    if args.use_vosk:
        vosk_entries = [v.strip() for v in args.vosk_models.split(",") if v.strip()]
        if not vosk_entries:
            raise ValueError("--use_vosk set but --vosk_models is empty.")
        for entry in vosk_entries:
            model_dir = ensure_vosk_model(entry, args.vosk_root)
            label = f"Vosk:{os.path.basename(model_dir)}"
            runners[label] = VoskRunner(model_dir=model_dir)

    model_names = list(runners.keys())
    if not model_names:
        raise ValueError("No models enabled. Use --whisper_models and/or --use_gemma_e2b/--use_gemma_e4b/--use_vosk")

    print(f"\nLoading LibriSpeech subset: split={args.split} n={args.num_samples} seed={args.seed} ...")
    ds = load_librispeech_subset(args.split, args.num_samples, args.seed)
    n = len(ds)

    print("\nModels to benchmark:")
    for mn in model_names:
        print(f" - {mn}")
    print()

    from jiwer import wer as jiwer_wer

    refs_raw: List[str] = []
    outputs_raw: Dict[str, List[str]] = {mn: [] for mn in model_names}
    latencies: Dict[str, List[float]] = {mn: [] for mn in model_names}

    records: List[Dict] = []

    for i, ex in enumerate(ds, 1):
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        ref = ex["text"]

        refs_raw.append(ref)
        ref_norm = normalize_for_wer(ref)

        row: Dict = {
            "sample_index": i,
            "split": args.split,
            "reference": ref,
            "sampling_rate": sr,
            "audio_num_samples": int(audio.shape[0]) if hasattr(audio, "shape") else None,
            "audio_duration_sec": (float(audio.shape[0]) / float(sr)) if hasattr(audio, "shape") else None,
        }

        for mn, runner in runners.items():
            t0 = time.perf_counter()
            try:
                hyp = runner.transcribe(audio, sr)
                err = None
            except Exception as e:
                hyp = ""
                err = repr(e)
            dt = time.perf_counter() - t0

            outputs_raw[mn].append(hyp)
            latencies[mn].append(dt)

            hyp_norm = normalize_for_wer(hyp)
            sample_wer = float(jiwer_wer([ref_norm], [hyp_norm]))

            col_prefix = (
                mn.replace(":", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
                .replace(" ", "_")
            )

            row[f"{col_prefix}_output"] = hyp
            row[f"{col_prefix}_latency_sec"] = dt
            row[f"{col_prefix}_wer"] = sample_wer
            if err:
                row[f"{col_prefix}_error"] = err

        records.append(row)

        if i % 10 == 0 or i == n:
            print(f"  processed {i}/{n}")

    refs = [normalize_for_wer(x) for x in refs_raw]
    per_sample_by_model: Dict[str, List[float]] = {}
    overall_by_model: Dict[str, float] = {}

    durations = np.array([r.get("audio_duration_sec") or np.nan for r in records], dtype=np.float64)

    rtf_by_model: Dict[str, List[float]] = {}
    for mn in model_names:
        lat = np.array(latencies[mn], dtype=np.float64)
        rtf = []
        for l, d in zip(lat, durations):
            if np.isfinite(l) and np.isfinite(d) and d > 0:
                rtf.append(float(l / d))
            else:
                rtf.append(float("nan"))
        rtf_by_model[mn] = rtf

    print("\nResults")
    print(f"Split: {args.split}")
    print(f"Samples: {n}")

    for mn in model_names:
        hyps = [normalize_for_wer(x) for x in outputs_raw[mn]]
        overall = compute_wer(refs, hyps)
        per_sample = compute_per_sample_wer(refs, hyps)

        overall_by_model[mn] = overall
        per_sample_by_model[mn] = per_sample

        lat = np.array(latencies[mn], dtype=np.float64)
        p50 = float(np.percentile(lat, 50))
        p90 = float(np.percentile(lat, 90))
        mean = float(lat.mean())

        durs = np.array([r.get("audio_duration_sec") or np.nan for r in records], dtype=np.float64)
        valid = np.isfinite(durs) & (durs > 0)
        rtf = (lat[valid] / durs[valid]) if valid.any() else np.array([], dtype=np.float64)
        rtf_mean = float(rtf.mean()) if rtf.size else float("nan")

        print(f"\nModel: {mn}")
        print(f"WER: {overall:.4f}")
        print(f"Latency (sec): mean={mean:.3f}  p50={p50:.3f}  p90={p90:.3f}")
        print(f"RTF (latency/audio): mean={rtf_mean:.3f}" if np.isfinite(rtf_mean) else "  RTF: n/a")

    title = f"LibriSpeech subset WER (split={args.split}, n={n})"
    plot_and_save_wer(per_sample_by_model, overall_by_model, args.plot_path, title)
    print(f"\nSaved WER plot to: {args.plot_path}")

    lat_title = f"LibriSpeech subset latency (split={args.split}, n={n})"
    plot_and_save_latency(latencies, args.latency_plot_path, lat_title, yscale=args.latency_scale)
    print(f"Saved latency plot to: {args.latency_plot_path}")

    if args.rtf_plot_path:
        rtf_title = f"LibriSpeech subset RTF (split={args.split}, n={n})"
        plot_and_save_rtf(rtf_by_model, args.rtf_plot_path, rtf_title, yscale=args.latency_scale)
        print(f"Saved RTF plot to: {args.rtf_plot_path}")

        
    df = pd.DataFrame(records)
    df.to_csv(args.log_path, index=False)
    print(f"Saved per-sample CSV log to: {args.log_path}")

    if args.save_jsonl:
        save_jsonl(args.jsonl_path, records)
        print(f"Saved per-sample JSONL log to: {args.jsonl_path}")

if __name__ == "__main__":
    main()