# mmllms-at-the-edge
Code, data, and experiments for our paper "How Do MMLLMs Fit In at the Edge?". The repo is organized in five different subdirectories, corresponding to the five tasks considered in the article (`OC`, `ASR`, `IC`, `AR`, `HD`).

## Reproducibility
As we are using HuggingFace for downloading the MMLLMs, to successfully download the models (as you run the code), you need to have a [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens#) configured, and saved under `~/.cache/huggingface/token` on your machine. Subsequently, you need to accept the licence for each of the models. For convenience, once you are logged into your HuggingFace account, simply navigate to each of the following links, and accept the terms and conditions for using the particular models:
1. `gemma3n-e2b`: [https://huggingface.co/google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)
2. `gemma3n-e4b`: [https://huggingface.co/google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)
3. `blip2`: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
4. `qwen3-vl-2b`: [https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
5. `ministral-3b`: [https://huggingface.co/mistralai/Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)

For the classic edge models, we either used `torchvision` models which should be already installed in the virtual environments, or alternatively, we provide the models as a file directly for the corresponding use-case(s) (rest assured, all instructions are provided in the adjacent README.md files).

Assuming you have configured your HuggingFace token and accepted the licences for the models, to reproduce the experiments, simply follow either of the instructions below for the use-case(s) you'd like to try first.
### 1. `OC` (Object Counting)
Navigate to `code/1_object_counting` and follow the instructions in the `README.md` file.
### 2. `ASR` (Automatic Speech Recogntion)
Navigate to `code/2_automatic_speech_recognition` and follow the instructions in the `README.md` file.
### 3. `IC` (Image Classification)
Navigate to `code/3_image_classification` and follow the instructions in the `README.md` file.
### 4. `AR` (Action Recognition)
Navigate to `code/4_action_recognition` and follow the instructions in the `README.md` file.
### 5. `HD` (Hazard Detection)
Navigate to `code/5_hazard_detection` and follow the instructions in the `README.md` file.