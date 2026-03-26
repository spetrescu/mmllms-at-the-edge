## 2. `ASR` (Automatic Speech Recognition)
1. Create virtual environment using `venv`. Run:
    1. `python3 -m venv env-asr`
    2. `source env-asr/bin/activate`
2. Install dependencies. Assuming `env-asr` is active, run:
    1. `pip install -r requirements.txt`
3. Run ASR after you `cd src/` with:
```
python 1_automatic_speech_recognition_librispeech.py \
  --split test.clean --num_samples 2620 --seed 0 \
  --whisper_models tiny,base,small,medium,large-v3,large-v3-turbo \
  --use_gemma_e2b --use_gemma_e4b \
  --use_vosk --vosk_models small-en-us-0.15,en-us-0.22 \
  --latency_scale log \
  --log_path results_asr_log_2620.csv --save_jsonl
```
4. For visualizations (with the existing results from the paper `results_asr_log_2620.csv`):
```
python 2_visualization_automatic_speech_recognition_librispeech.py --csv results_asr_log_2620.csv --out_dir ./figs --latency_scale log
  ```
