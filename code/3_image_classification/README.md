## 3. `IC` (Image Classification)
1. Create virtual environment using `venv`. Run:
    1. `python3 -m venv env-ic`
    2. `source env-ic/bin/activate`
2. Install dependencies. Assuming `env-ic` is active, run:
    1. `pip install -r requirements.txt`
3. Run IC after you `cd src/` with:
```
python 1_image_classification_imagenette.py \
  --dataset frgfm/imagewoof --config 160px --split validation --num_samples 3925 --seed 0 \
  --tv_models mobilenet_v3_large,resnet50 \
  --use_qwen3_vl_2b --use_gemma_e4b --use_gemma_e2b --use_ministral3_3b
```
4. For visualizations (with the existing results from the paper `results_imgcls_log_3925_validation.csv`):
```
python 2_visualizations_image_classification_imagenette.py \
  --log_csv results_imgcls_log_3925_validation.csv \
  ```