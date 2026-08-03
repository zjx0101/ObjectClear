<div align="center">
<div style="text-align: center;">
    <img src="./assets/ObjectClear_logo.png" alt="ObjectClear Logo" style="height: 52px;">
    <h2>Precise Object and Effect Removal with Adaptive Target-Aware Attention</h2>
</div>

<div>
    <a href="https://zjx0101.github.io/" target='_blank'>Jixin Zhao<sup>*</sup></a>&emsp;
    <a href='https://wzhouxiff.github.io' target='_blank'>Zhouxia Wang</a>&emsp;
    <a href='https://pq-yang.github.io/' target='_blank'>Peiqing Yang</a>&emsp;
    <a href='https://shangchenzhou.com/' target='_blank'>Shangchen Zhou<sup>*,†</sup>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; 
</div>

<div>
    <strong>CVPR 2026 </strong>
</div>


<div>
    <h4 align="center">
        <a href="https://zjx0101.github.io/projects/ObjectClear/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2505.22636" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2505.22636-b31b1b.svg">
        </a>
        <a href="https://huggingface.co/spaces/jixin0101/ObjectClear" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue">
        </a>
        <a href="https://huggingface.co/datasets/sczhou/OBERDataset_ObjectClear" target='_blank'>
        <img src="https://img.shields.io/badge/Dataset-%F0%9F%A7%B8%20OBER-blue">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/ObjectClear&ltext=Visitors&color=3977dd">
    </h4>
</div>

<strong>ObjectClear is an object removal model that can jointly eliminate the target object and its associated effects leveraging Adaptive Target-Aware Attention, while preserving background consistency.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/teaser.png">
</div>

For more visual results, go checkout our <a href="https://zjx0101.github.io/projects/ObjectClear/" target="_blank">project page</a>

---
</div>


## ⭐ Update
- [2026.02] **🔥 Training code is now released!** See the [Training](#-training) section below.
- [2026.02] **🔥 OBER Dataset is Now Released!** Our training dataset is now publicly available on [Hugging Face](https://huggingface.co/datasets/sczhou/OBERDataset_ObjectClear) 🤗.
- [2025.09] We have released our [benchmark datasets](https://drive.google.com/drive/folders/12LA53ZPAG1uxdVXsn90L2qe6zCcp6aGF?usp=sharing) for evaluation, along with [our results](https://drive.google.com/drive/folders/1eUbIz5OS9yK6Ih8Y1qXoXuk_UWOcifcY?usp=sharing) to facilitate comparison.
- [2025.07] Release the inference code and Gradio demo.
- [2025.05] This repo is created.

### ✅ TODO
- [x] Release the training code
- [x] Release our training datasets
- [x] Release our benchmark datasets
- [x] Release the inference code and Gradio demo


## 🎃 Overview
![overall_structure](assets/ObjectClear_pipeline.png)


## 📷 OBER Dataset
![OBER_dataset_pipeline](assets/OBER_pipeline.png)

OBER (OBject-Effect Removal) is a hybrid dataset designed to support research in object removal with effects, combining both camera-captured and simulated data. 

🔥 We have released the full dataset [OBERDataset_ObjectClear](https://huggingface.co/datasets/sczhou/OBERDataset_ObjectClear ) on Hugging Face. We hope it can serve as a strong training resource and benchmark for future object removal research.

> 🚩 Note that the OBER dataset are made available solely for **non-commercial** research use. Any use, reproduction, or redistribution must strictly comply with the terms of <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>.

![OBER_dataset_samples](assets/dataset_samples.png)



## ⚙️ Installation
1. Clone Repo
    ```bash
    git clone https://github.com/zjx0101/ObjectClear.git
    cd ObjectClear
    ```

2. Create Conda Environment and Install Dependencies
    ```bash
    # create new conda env
    conda create -n objectclear python=3.10 -y
    conda activate objectclear

    # install python dependencies
    pip3 install -r requirements.txt
    # [optional] install python dependencies for gradio demo
    pip3 install -r hugging_face/requirements.txt
    ```


## ⚡ Inference

### Quick Test
We provide some examples in the [`inputs`](./inputs) folder. **For each run, we take an image and its segmenatation mask as input.** <u>The segmentation mask can be obtained from interactive segmentation models such as [SAM2 demo](https://huggingface.co/spaces/fffiloni/SAM2-Image-Predictor)</u>. For example, the directory structure can be arranged as follows:
```
inputs
   ├─ imgs
   │   ├─ test-sample1.jpg      # .jpg, .png, .jpeg supported
   │   ├─ test-sample2.jpg
   └─ masks
       ├─ test-sample1.png
       ├─ test-sample2.png
```
Run the following command to try it out:

```shell
## Single image inference
python inference_objectclear.py -i inputs/imgs/test-sample1.jpg -m inputs/masks/test-sample1.png --guidance_scale 2.5 --use_fp16

## Batch inference on image folder
python inference_objectclear.py -i inputs/imgs -m inputs/masks --guidance_scale 2.5 --use_fp16
```

> **Note:** `--guidance_scale` controls the trade-off: higher values lead to stronger removal, while lower values better preserve background details.  
> The default setting is `--guidance_scale 2.5`. For all [benchmark results](https://drive.google.com/drive/folders/1eUbIz5OS9yK6Ih8Y1qXoXuk_UWOcifcY?usp=sharing) reported in our paper, we used `--guidance_scale 1.0`.


## 🚀 Training

### 1. Prepare the pretrained weights
ObjectClear is built on [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) and uses a [CLIP image encoder](https://huggingface.co/openai/clip-vit-large-patch14) to encode the target object. The SDXL base model is downloaded automatically on the first run. Download the CLIP image encoder into `./ckpts`:

```shell
# download the CLIP image encoder used as the object encoder
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/clip-vit-large-patch14
```

### 2. Prepare the OBER dataset
The training reads the OBER parquet shards directly (no unpacking needed). Access to the dataset is gated — first request access on the [dataset page](https://huggingface.co/datasets/sczhou/OBERDataset_ObjectClear), then log in and download:

```shell
# log in with your Hugging Face token (needed for the gated dataset)
huggingface-cli login

# download the OBER dataset (~27 GB) into ./data/OBER
huggingface-cli download sczhou/OBERDataset_ObjectClear --repo-type dataset --local-dir ./data/OBER
```

After downloading, the parquet shards live in `./data/OBER/data`:
```
data/OBER/data
   ├─ train-00000-of-00053.parquet   # 37,994 cropped training pairs
   ├─ ...
   ├─ train-00052-of-00053.parquet
   └─ test-00000-of-00001.parquet    # test split (used for validation)
```
Each sample provides `input`, `gt`, `object_mask`, and `object_effect_mask`.

### 3. Start training
We provide a ready-to-run script [`train.sh`](./train.sh) for multi-GPU training with 🤗 `accelerate`. Edit the paths / hyper-parameters at the top of the script, then run:

```shell
bash train.sh
```

Or launch directly with `accelerate` (8 GPUs example):
```shell
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 \
    train_objectclear.py \
    --pretrained_model_name_or_path "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
    --image_encoder_name_or_path "./ckpts/clip-vit-large-patch14" \
    --output_dir "./runs/train_objectclear" \
    --image_dir1 "./data/OBER/data" \
    --resolution 512 \
    --train_batch_size 4 \
    --learning_rate 1e-05 \
    --learning_rate_attn 1e-05 \
    --lr_scheduler cosine \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --color_augmentation \
    --flip_augmentation \
    --random_mask_dilation \
    --random_mask_erosion \
    --object_localization \
    --object_localization_weight 0.01 \
    --background_loss_weight 1 \
    --real_only \
    --seed 42
```

### 4. Validation during training
To monitor training, enable validation on the OBER test split. Validation runs the full [`ObjectClearPipeline`](./objectclear/pipelines) (identical to inference) and reports PSNR on samples with ground truth:

```shell
    --validation_parquet "./data/OBER/data/test-00000-of-00001.parquet" \
    --validation_subset "OBER-Test" \
    --validation_num_samples 8 \
    --validate_by_iter \
    --validation_iterations 2000
```

> **Note:** `--validation_subset` can be `OBER-Test` or `RORD-Val-343` (both have ground truth, so PSNR is computed) or `OBER-Wild` (no ground truth). Validation images, attention maps, and metrics are written to `<output_dir>/validation_results/`.


## 📊 Evaluation with ReMOVE+
Our **ReMOVE+** metric addresses the limitations of the original ReMOVE by assessing consistency between the output's object-effect region and the input's background (outside the object-effect mask), making it more suitable for object-effect removal evaluation.

Please refer to the detailed instructions in the [`evaluation/README.md`](./evaluation/README.md) file for installation, setup, and running the ReMOVE+ evaluation pipeline.



## 🪄 Interactive Demo
To get rid of the preparation for segmentation mask, we prepare a gradio demo on [hugging face](https://huggingface.co/spaces/jixin0101/ObjectClear) and could also [launch locally](./hugging_face). Just drop your image, assign the target masks with a few clicks, and get the object removal results!
```shell
cd hugging_face

# install python dependencies
pip3 install -r requirements.txt

# launch the demo
python app.py
```

<p align="center">
  <img src="assets/user_clicks.gif" width="49%" />
  <img src="assets/user_strokes.gif" width="49%" />
</p>


## 📝 License
**Non-Commercial Use Only Declaration**

The ObjectClear is made available for use, reproduction, and distribution strictly for non-commercial purposes. The code, models, and datasets are licensed under <a rel="license" href="./LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.



## 📑 Citation
If you find our repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{zhao2026objectclear,
    title   = {Precise Object and Effect Removal with Adaptive Target-Aware Attention},
    author  = {Zhao, Jixin and Wang, Zhouxia and Yang, Peiqing and Zhou, Shangchen},
    booktitle = {CVPR},
    year    = {2026},
    }
```

## 📧 Contact
If you have any questions, please feel free to reach us at `jixinzhao0101@gmail.com` and `shangchenzhou@gmail.com`. 
