## 4. `AR` (Action Recognition)
1. Create virtual environment using `venv`. Run:
    1. `python3 -m venv env-ar`
    2. `source env-ar/bin/activate`
2. Install dependencies. Assuming `env-ar` is active, run: `pip install -r requirements.txt`
3. Run AR after you `cd src/` with:
```
python 1_action_recognition_kinetics.py \
  --dataset nateraw/kinetics-mini --split validation --num_samples 50 --seed 0 \
  --tv_models r3d_18,mc3_18,r2plus1d_18 \
  --use_gemma_e2b --use_gemma_e4b \
  --device auto \
  --clip_len 16 --stride 4 \
  --gemma_frames 8 --grid_cols 4 \
  --latency_scale log \
  --latency_plot_path ar_latency_kinetics.png --acc_plot_path ar_accuracy_kinetics.png \
  --log_path ar_kinetics_log.csv --save_jsonl
```
4. For visualizations (with the existing results from the paper `ar_kinetics_log.csv`):
```
python 2_visualizations_action_recognition_kinetics.py \
  --log_csv ar_kinetics_log.csv
```