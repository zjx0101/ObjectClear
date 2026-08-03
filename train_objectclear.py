"""Training script for ObjectClear.

Example (8 GPUs). See train.sh for a ready-to-run wrapper and the README for
dataset / checkpoint preparation:

accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 \
train_objectclear.py \
    --pretrained_model_name_or_path "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
    --image_encoder_name_or_path "./ckpts/clip-vit-large-patch14" \
    --output_dir "./runs/train_objectclear" \
    --image_dir1 "./data/OBER/data" \
    --validation_parquet "./data/OBER/data/test-00000-of-00001.parquet" \
    --validation_subset "OBER-Test" \
    --validate_by_iter \
    --validation_iterations 2000 \
    --train_batch_size 4 \
    --learning_rate 1e-05 \
    --learning_rate_attn 1e-05 \
    --resolution 512 \
    --max_train_steps 100000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --color_augmentation \
    --flip_augmentation \
    --random_mask_dilation \
    --random_mask_erosion \
    --lr_scheduler cosine \
    --background_loss_weight 1 \
    --gradient_accumulation_steps 1 \
    --object_localization \
    --object_localization_weight 0.01 \
    --real_only \
    --seed 42
"""
import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch.utils.data as data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from objectclear.models import CLIPImageEncoder, PostfuseModule
import types
import copy

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from objectclear.dataset.objectclear_dataset import ObjectClearDataset
from objectclear.dataset.ober_parquet_dataset import OBERParquetDataset
from objectclear.pipelines import ObjectClearPipeline

if is_wandb_available():
    import wandb

check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def save_train_argument_param_yaml(output_dir, args, script_path=None):
    """Save the training arguments to a YAML file for provenance.

    Inlined replacement for the former external ``train_run_config`` module.
    """
    import yaml

    os.makedirs(output_dir, exist_ok=True)
    record = {"args": vars(args)}
    if script_path is not None:
        record["script"] = os.path.abspath(script_path)
    record["saved_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = os.path.join(output_dir, "train_config.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, allow_unicode=True, sort_keys=False)
    logger.info(f"Saved training config to {out_path}")


def compute_psnr(pred: Image.Image, gt: Image.Image) -> float:
    """PSNR (dB) between two RGB PIL images, computed in torch. Zero extra deps."""
    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.BICUBIC)
    pred_t = torch.from_numpy(np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0)
    gt_t = torch.from_numpy(np.asarray(gt.convert("RGB"), dtype=np.float32) / 255.0)
    mse = torch.mean((pred_t - gt_t) ** 2)
    if mse.item() <= 1e-10:
        return 100.0
    return float(10.0 * torch.log10(1.0 / mse).item())


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision,
        cache_dir=args.model_dir,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validate_by_iter",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--validation_iterations",
        type=int,
        default=100,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Legacy jsonl index for validation images. Leave unset to disable "
        "validation, or use --validation_parquet to validate from OBER parquet.",
    )
    parser.add_argument(
        "--validation_parquet",
        type=str,
        default=None,
        help="Path to the OBER test parquet (e.g. ./data/OBER/data/test-00000-of-00001.parquet). "
        "When set, validation reads images directly from it.",
    )
    parser.add_argument(
        "--validation_subset",
        type=str,
        default="OBER-Test",
        help="Which subset of the validation parquet to use. OBER-Test and "
        "RORD-Val-343 have GT (PSNR computed); OBER-Wild has no GT.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="If set, only run validation on the first N samples of the subset.",
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--object_localization",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--object_localization_weight", type=float, default=1.0, help="The weight of localization loss.")
    parser.add_argument("--background_loss_weight", type=float, default=10.0, help="The weight of background loss in localization loss calculation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--triple_reflection_dataset",
        action="store_true",
        help="whether to triple the reflection dataset",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_postfuse_module_only",
        action="store_true",
        help="Whether to only train the postfuse module.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_attn",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--prediction_type", type=str, default=None) 
    # Our added arguments
    parser.add_argument(
        "--real_only",
        action='store_true',
        help="Only train with real dataset."
    )
    parser.add_argument(
        "--image_dir1",
        type=str,
        default="",
        help="The dir of input images 1.",
    )
    parser.add_argument(
        "--image_dir2",
        type=str,
        default="",
        help="The dir of input images 2.",
    )
    parser.add_argument(
        "--image_dir3",
        type=str,
        default="",
        help="The dir of input images 3.",
    )
    parser.add_argument(
        "--image_dir4",
        type=str,
        default="",
        help="The dir of input images 4.",
    )
    parser.add_argument(
        "--is_middle_crop",
        action='store_true',
        help="Enable middle crop."
    )
    parser.add_argument(
        "--use_blank_mask",
        action='store_true',
        help="Add blank mask to the training set."
    )
    parser.add_argument(
        "--color_augmentation",
        action='store_true',
        help="Use color augmentation or not."
    )
    parser.add_argument(
        "--flip_augmentation",
        action='store_true',
        help="Use flip augmentation or not."
    )
    parser.add_argument(
        "--random_mask_dilation",
        action='store_true',
        help="Use random mask dilation or not."
    )
    parser.add_argument(
        "--random_mask_erosion",
        action='store_true',
        help="Use random mask dilation or not."
    )
    parser.add_argument(
        "--no_structural_mask_augment",
        action="store_true",
        help=(
            "Disable SmartEraser-style structural mask augmentation on object and shadow masks. "
            "When set, use legacy random dilation/erosion on the object mask only."
        ),
    )
    parser.add_argument(
        "--mask_weight",
        type=float,
        default=None,
        help=(
            "The loss weight of the masked area"
        ),
    )
    parser.add_argument(
        "--image_add_mask",
        action="store_true",
        help="Mask the original image as input."
    )
    parser.add_argument(
        "--add_mask_to_shadow_mask",
        action="store_true",
        help="Mask the original image as input."
    )
    parser.add_argument(
        "--input_blending",
        action="store_true",
        help="Mask the original image as input."
    )
    parser.add_argument(
        "--reverse_shadow_mask",
        action="store_true",
        help="Reverse the shadow mask."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model_cache",
        help="The dir of models",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="The dir of pretrained path for continuing training",
    )
    parser.add_argument(
        "--pretrained_path_postfuse",
        type=str,
        default=None,
        help="The dir of pretrained path for continuing training",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # if args.dataset_name is None and args.instance_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    # if args.dataset_name is not None and args.instance_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.0",
        "up_blocks.1",
        "up_blocks.2",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            # print(f"Unet layer name: {name}, module: {module}")
            if not any(layer in name for layer in applicable_layers):
                continue
            # print(f"Unet applicable layer name: {name}, module: {module}")
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.new_get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
            module.get_attention_scores = module.new_get_attention_scores

    return unet


def toggle_unet_attention_scores(unet, enable_storage=True):
    from diffusers.models.attention_processor import Attention

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and 'attn2' in name:
            if enable_storage:
                # Swap in the patched method that stores cross-attention scores.
                if hasattr(module, 'new_get_attention_scores'):
                    module.get_attention_scores = module.new_get_attention_scores
            else:
                # Restore the original get_attention_scores method.
                if hasattr(module, 'old_get_attention_scores'):
                    module.get_attention_scores = module.old_get_attention_scores


def clear_cross_attention_scores(cross_attention_scores):
    keys = list(cross_attention_scores.keys())
    for k in keys:
        del cross_attention_scores[k]
    gc.collect()

def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    loss_fn,
    fuse_index,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )
    
    index_tensor = torch.full(
        (b, num_heads, num_noise_latents, 1), fuse_index, dtype=torch.long, device=cross_attention_scores.device
    )

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(cross_attention_scores, dim=3, index=index_tensor)  # (b, num_heads, num_noise_latents, max_num_objects)
    object_segmaps = object_segmaps.permute(0, 2, 1).unsqueeze(1).expand(b, num_heads, num_noise_latents, 1)
    loss = loss_fn(object_token_attn_prob, object_segmaps)
    loss = loss.mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    loss_fn,
    fuse_index,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, loss_fn, fuse_index
        )
        loss += layer_loss
    return loss / num_layers


class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=0.1, normalize=False, background_loss_weight=1):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize
        self.background_loss_weight = background_loss_weight

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        object_segmaps = (object_segmaps > self.threshold).to(object_segmaps.dtype)
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return self.background_loss_weight * background_loss - object_loss + 1


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def _hstack_images_horiz(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    w1, h = left.size
    w2, h2 = right.size
    if (w2, h2) != (w1, h):
        right = right.resize((w1, h), Image.BICUBIC)
    out = Image.new("RGB", (w1 + w2, h))
    out.paste(left, (0, 0))
    out.paste(right, (w1, 0))
    return out


def _val_asset(rel_or_abs: str, val_dir: str) -> str:
    p = rel_or_abs.strip()
    return p if os.path.isabs(p) else os.path.join(val_dir, p)


def _validation_enabled(args) -> bool:
    return args.validation_file is not None or args.validation_parquet is not None


def _load_validation_examples_from_parquet(args):
    """Yield uniform validation examples from an OBER test parquet.

    Each example is a dict with PIL images:
        {img_name, orig_image_rgb, gt_image (or None), mask_full, shadow_mask_full}
    Field mapping matches the training dataset: input / gt / object_mask /
    object_effect_mask. OBER-Wild has no GT, so gt_image is None there.
    """
    import io
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.validation_parquet)
    examples = []
    count = 0
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(
            rg, columns=["subset", "input", "gt", "object_mask", "object_effect_mask"]
        )
        n = table.num_rows
        for i in range(n):
            if table["subset"][i].as_py() != args.validation_subset:
                continue

            def _decode(col, mode):
                cell = table[col][i].as_py()
                b = cell.get("bytes") if isinstance(cell, dict) else None
                if not b:
                    return None, None
                im = Image.open(io.BytesIO(b))
                path = cell.get("path") if isinstance(cell, dict) else None
                return im.convert(mode), path

            orig_image_rgb, path = _decode("input", "RGB")
            if orig_image_rgb is None:
                continue
            gt_image, _ = _decode("gt", "RGB")
            mask_full, _ = _decode("object_mask", "L")
            shadow_mask_full, _ = _decode("object_effect_mask", "L")

            img_name = path or f"{args.validation_subset}_{count}.png"
            examples.append(
                {
                    "img_name": os.path.basename(img_name),
                    "orig_image_rgb": orig_image_rgb,
                    "gt_image": gt_image,
                    "mask_full": mask_full,
                    "shadow_mask_full": shadow_mask_full,
                }
            )
            count += 1
            if args.validation_num_samples is not None and count >= args.validation_num_samples:
                return examples
    return examples


def _validation_example_is_ober_style(val_example: dict) -> bool:
    return "img" in val_example and "mask" in val_example


def _try_load_gt_image(
    val_example: dict, val_dir: str, ref_basename: str
) -> Optional[Image.Image]:
    if "gt" in val_example:
        return Image.open(_val_asset(val_example["gt"], val_dir)).convert("RGB")
    gfn = val_example.get("gt_file_name")
    if gfn:
        p = _val_asset(gfn, val_dir)
        if os.path.isfile(p):
            return Image.open(p).convert("RGB")
    cand = os.path.join(val_dir, "gt", ref_basename)
    if os.path.isfile(cand):
        return Image.open(cand).convert("RGB")
    return None


def validate(args, epoch, accelerator, unet, postfuse_module, image_encoder, weight_dtype, global_step):
    unet_unwrapped = accelerator.unwrap_model(copy.deepcopy(unet)).float()
    postfuse_module_unwrapped = accelerator.unwrap_model(copy.deepcopy(postfuse_module)).float()
    image_encoder_unwrapped = accelerator.unwrap_model(copy.deepcopy(image_encoder)).float()
    toggle_unet_attention_scores(unet_unwrapped, enable_storage=False)
    from_parquet = args.validation_parquet is not None
    if from_parquet:
        val_dir = args.output_dir
        val_examples = _load_validation_examples_from_parquet(args)
    else:
        val_dir = os.path.dirname(args.validation_file)
        val_examples = []
        with open(args.validation_file, "r") as val_index_fp:
            for line in val_index_fp:
                val_examples.append(json.loads(line))

    logger.info(
        f"Running validation... \n Generating {len(val_examples)} images"
    )
    # Create the pipeline. Use ObjectClearPipeline so that validation matches
    # inference exactly. AGF is disabled here (raw diffusion output), but we still
    # request the attention map via return_attn_map for visualisation.
    # The unwrapped modules above are cast to float() (fp32). Load the rest of the
    # pipeline (vae, text encoders) in fp32 too so all modules share one dtype;
    # mixing fp16 weights with the fp32 unet triggers dtype-mismatch errors.
    pipeline = ObjectClearPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet_unwrapped,
        postfuse_module=postfuse_module_unwrapped,
        image_prompt_encoder=image_encoder_unwrapped,
        revision=args.revision,
        torch_dtype=torch.float32,
        apply_attention_guided_fusion=False,
        cache_dir=args.model_dir,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    attn_map_list = []
    save_dir = os.path.join(args.output_dir, "validation_results", f"step_{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    # Only PSNR is computed for now (pure torch, no extra deps).
    # TODO: add LPIPS here once the `lpips` package is available in the env.
    metrics_all: list = []

    for val_example in tqdm(val_examples):
        if from_parquet:
            img_name = val_example["img_name"]
            orig_image_rgb = val_example["orig_image_rgb"]
            orig_size = orig_image_rgb.size
            gt_image = val_example["gt_image"]
            if gt_image is not None and gt_image.size != orig_size:
                gt_image = gt_image.resize(orig_size, Image.BILINEAR)
            mask_full = val_example["mask_full"]
            if mask_full.size != orig_size:
                mask_full = mask_full.resize(orig_size, Image.NEAREST)
            shadow_mask_full = val_example["shadow_mask_full"]
            if shadow_mask_full.size != orig_size:
                shadow_mask_full = shadow_mask_full.resize(orig_size, Image.NEAREST)
        elif _validation_example_is_ober_style(val_example):
            img_name = os.path.basename(val_example["img"])
            orig_image_rgb = Image.open(_val_asset(val_example["img"], val_dir)).convert("RGB")
            orig_size = orig_image_rgb.size
            gt_image = Image.open(_val_asset(val_example["gt"], val_dir)).convert("RGB")
            if gt_image.size != orig_size:
                gt_image = gt_image.resize(orig_size, Image.BILINEAR)
            mask_full = Image.open(_val_asset(val_example["mask"], val_dir)).convert("L")
            if mask_full.size != orig_size:
                mask_full = mask_full.resize(orig_size, Image.NEAREST)
            shadow_mask_full = Image.open(_val_asset(val_example["effect_mask"], val_dir)).convert("L")
            if shadow_mask_full.size != orig_size:
                shadow_mask_full = shadow_mask_full.resize(orig_size, Image.NEAREST)
        else:
            img_name = os.path.basename(val_example["file_name"])
            orig_image_rgb = Image.open(os.path.join(val_dir, val_example["file_name"])).convert("RGB")
            orig_size = orig_image_rgb.size
            gt_image_opt = _try_load_gt_image(val_example, val_dir, img_name)
            gt_image = gt_image_opt
            mask_full = (
                Image.open(os.path.join(val_dir, val_example["mask_file_name"]))
                .convert("L")
            )
            if mask_full.size != orig_size:
                mask_full = mask_full.resize(orig_size, Image.NEAREST)
            shadow_mask_full = (
                Image.open(os.path.join(val_dir, val_example["shadow_mask_file_name"]))
                .convert("L")
            )
            if shadow_mask_full.size != orig_size:
                shadow_mask_full = shadow_mask_full.resize(orig_size, Image.NEAREST)
            if gt_image is not None and gt_image.size != orig_size:
                gt_image = gt_image.resize(orig_size, Image.BILINEAR)

        val_image = orig_image_rgb.resize(
            (args.resolution, args.resolution), Image.BILINEAR
        )
        mask_image = mask_full.resize(
            (args.resolution, args.resolution), Image.NEAREST
        )
        shadow_mask_image = shadow_mask_full.resize(
            (args.resolution, args.resolution), Image.NEAREST
        )
        if args.input_blending and global_step < 2000:
            mask_image_np = np.array(mask_image)
            shadow_mask_image_np = np.array(shadow_mask_image)
            val_image_np = np.array(val_image)

            alpha = global_step / 2000.0
            blended_mask = alpha * mask_image_np + (1 - alpha) * shadow_mask_image_np

            blended_mask_np = 1 - blended_mask / 255.0
            blended_mask_np = blended_mask_np[:, :, None]
            blended_mask_np = np.repeat(blended_mask_np, 3, axis=2)

            masked_val_image_np = val_image_np.copy()
            masked_val_image_np = (masked_val_image_np / 255.0 * 2 - 1) * blended_mask_np
            masked_val_image_np = (masked_val_image_np + 1) / 2 * 255

            masked_val_image_np = (global_step / 2000) * val_image_np + (
                1 - global_step / 2000
            ) * masked_val_image_np
            val_image = Image.fromarray(masked_val_image_np.astype(np.uint8))
            mask_image = Image.fromarray(blended_mask.astype(np.uint8))
        images.append(val_image)
        result = pipeline(
            prompt="remove the instance of object",
            image=val_image,
            mask_image=mask_image,
            num_inference_steps=20,
            strength=0.99,
            generator=generator,
            guidance_scale=3.5,
            height=args.resolution,
            width=args.resolution,
            return_attn_map=True,
        )
        output = result.images[0]
        images.append(output)

        pred_rgb = output.convert("RGB")
        pred_up = pred_rgb.resize(orig_size, Image.BICUBIC)
        vis_pair = _hstack_images_horiz(orig_image_rgb, pred_up)
        vis_pair.save(os.path.join(save_dir, img_name))

        if accelerator.is_main_process and gt_image is not None:
            try:
                psnr = compute_psnr(pred_up, gt_image)
                metrics_all.append({"stem": os.path.splitext(img_name)[0], "psnr": psnr})
            except Exception as ex:
                logger.warning("validation PSNR skipped for %s: %s", img_name, ex)

        # Attention maps come directly from ObjectClearPipeline (return_attn_map).
        attns = getattr(result, "attns", None)
        if attns:
            attn_map_list.append(attns[0].resize((512, 512)))

    if accelerator.is_main_process and metrics_all:
        mean_psnr = float(np.mean([m["psnr"] for m in metrics_all]))
        meta = {
            "step": int(global_step),
            "epoch": int(epoch),
            "validation_file": args.validation_file,
            "n": len(metrics_all),
            "source": "train_objectclear validate; ObjectClearPipeline (AGF off); PSNR only",
        }
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"meta": meta, "per_image": metrics_all, "mean": {"psnr": mean_psnr}},
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("validation metrics: PSNR=%.4f (n=%d)", mean_psnr, len(metrics_all))
        logger.info("validation metrics saved to %s", metrics_path)

        summary_jsonl = os.path.join(
            args.output_dir, "validation_metrics_summary.jsonl"
        )
        recorded_at = datetime.now().isoformat(timespec="milliseconds")
        summary_record = {
            "recorded_at": recorded_at,
            "step": int(global_step),
            "epoch": int(epoch),
            "n": len(metrics_all),
            "mean": {"psnr": mean_psnr},
        }
        with open(summary_jsonl, "a", encoding="utf-8") as sf:
            sf.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        logger.info("validation metrics summary appended to %s", summary_jsonl)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, global_step, dataformats="NHWC"
            )
            if attn_map_list:
                np_attn_map = np.stack(
                    [
                        np.asarray(attn_map.convert("L")).reshape(512, 512, 1)
                        for attn_map in attn_map_list
                    ]
                )
                tracker.writer.add_images(
                    "attn_map", np_attn_map, global_step, dataformats="NHWC"
                )
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(
                            image, caption=f"{i}: {val_example.get('text', '')}"
                        )
                        for i, (val_example, image) in enumerate(
                            zip(val_examples, images)
                        )
                    ]
                }
            )
    toggle_unet_attention_scores(unet_unwrapped, enable_storage=True)
    del unet_unwrapped
    del pipeline
    torch.cuda.empty_cache()


def _save_accelerator_checkpoint(accelerator, args, global_step):
    """Mirror periodic checkpoint save (prune + save_state). Main process only; caller should sync processes."""
    if not accelerator.is_main_process or args.output_dir is None or global_step < 1:
        return
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir)
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            save_train_argument_param_yaml(
                args.output_dir, args, script_path=__file__
            )

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
        cache_dir=args.model_dir,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
        cache_dir=args.model_dir,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    scheduler_type = determine_scheduler_type(args.pretrained_model_name_or_path, args.revision)
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.model_dir)
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler",
            cache_dir=args.model_dir,
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.model_dir)

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, cache_dir=args.model_dir
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, cache_dir=args.model_dir
    )
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.model_dir,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.model_dir
    )
    
    cross_attention_scores = {}
    if args.object_localization:
        unet = unet_store_cross_attention_scores(
            unet, cross_attention_scores, layers=5
        )
        object_localization_loss_fn = BalancedL1Loss(threshold=0.1, background_loss_weight=args.background_loss_weight)
    
    image_encoder = CLIPImageEncoder.from_pretrained(
        args.image_encoder_name_or_path,
        cache_dir=args.model_dir,
        subfolder="",
    )
    postfuse_module = PostfuseModule(embed_dim=2048, embed_dim_img=768)
    
    if args.pretrained_path:
        state_dict = load_file(args.pretrained_path)
        print(f'loading from {args.pretrained_path}')
        unet.load_state_dict(state_dict)
    if args.pretrained_path_postfuse:
        state_dict = load_file(args.pretrained_path_postfuse)
        print(f'loading from {args.pretrained_path_postfuse}')
        postfuse_module.load_state_dict(state_dict)

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    if args.train_postfuse_module_only:
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    postfuse_module.requires_grad_(True)
    image_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            print(f'Using xFormers. the xFormers version is {xformers_version}.')
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_attn = (
            args.learning_rate_attn * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    unet_parameters = unet.parameters()
    postfuse_module_parameters = postfuse_module.parameters()
    text_parameters_one = text_encoder_one.parameters()
    text_parameters_two = text_encoder_two.parameters()
    unet_parameters_with_lr = {"params": unet_parameters, "lr": args.learning_rate}
    postfuse_module_parameters_with_lr = {"params": postfuse_module_parameters, "lr": args.learning_rate_attn}
    params_to_optimize = [unet_parameters_with_lr, postfuse_module_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params = (itertools.chain(unet.parameters(), postfuse_module.parameters()))
        optimizer = optimizer_class(
            params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    train_dataset1 = OBERParquetDataset(parquet_dir=args.image_dir1, input_size=args.resolution, is_middle_crop=args.is_middle_crop, \
        use_blank_mask=args.use_blank_mask, color_augmentation=args.color_augmentation, flip_augmentation=args.flip_augmentation, random_mask_dilation=args.random_mask_dilation, random_mask_erosion=args.random_mask_erosion, \
        structural_mask_augment=not args.no_structural_mask_augment)

    train_dataset = train_dataset1
    train_dataset.custom_instance_prompts = False

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt = 'remove the instance of object'
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            instance_prompt, text_encoders, tokenizers
        )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt)
            tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  #38000/4=9500
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  #1* 9500=9500
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet.train()
    postfuse_module.train()
    # Prepare everything with our `accelerator`.
    unet, postfuse_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, postfuse_module, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = (
            "sd-xl"
            if "playground" not in args.pretrained_model_name_or_path
            else "playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # validate(args, -1, accelerator, unet, postfuse_module, image_encoder, weight_dtype, global_step)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        postfuse_module.train()
        image_encoder.eval()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet,postfuse_module):
                clear_cross_attention_scores(cross_attention_scores)
                pixel_values = batch["gt"].to(dtype=vae.dtype)
                prompts = batch["prompt"]
                obj_only = batch['obj_only_tensor']
                shadow_mask = batch["shadow_mask"]
                masks =  batch["mask"]
                if args.add_mask_to_shadow_mask:
                    shadow_mask = torch.max(masks, shadow_mask)
                if args.reverse_shadow_mask:
                    shadow_mask = 1 - shadow_mask

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts)
                        
                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Convert masked images to latent space
                if args.image_add_mask:
                    masked_latents = vae.encode(
                        batch["masked_image"].reshape(batch["gt"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                elif args.input_blending and global_step<2000:
                    alpha = global_step / 2000
                    masks = ((1 - alpha) * shadow_mask  + alpha * masks)
                    input_image = (global_step / 2000) * batch["input"] + (1 - global_step / 2000) * (1 - masks) * batch["input"]
                    masked_latents = vae.encode(
                        input_image.reshape(batch["gt"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                else:
                    masked_latents = vae.encode(
                        batch["input"].reshape(batch["gt"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                
                mask = torch.nn.functional.interpolate(
                    batch["mask"],
                    size=(
                        args.resolution // 8,
                        args.resolution // 8,
                    ),
                )
                mask = mask.to(device=model_input.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                if not args.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
                # We then precondition the final model inputs based on these sigmas instead of the timesteps.
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                if args.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=(args.resolution, args.resolution), crops_coords_top_left=(0, 0))
                        for s in batch['input']
                    ]
                )

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = bsz
                else:
                    elems_to_repeat_text_embeds = 1

                # concatenate the noised latents with the mask and the masked latents
                noisy_model_input = torch.cat([noisy_model_input, mask, masked_latents], dim=1)

                # Predict the noise residual
                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                }
                repeated_prompt_embeds = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                object_embeds = image_encoder(obj_only)
                fuse_index = 5
                prompt_embeds_input = postfuse_module(repeated_prompt_embeds, object_embeds, fuse_index)
                model_pred = unet(
                    inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]
                    
                weighting = None
                if args.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas**2 + 1)
                            )
                    # We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if args.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if args.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        if args.mask_weight is not None:
                            mask_expanded = mask.expand_as(model_pred)
                            weight_inside_mask = args.mask_weight
                            weight_outside_mask = 1
                            weight_matrix = torch.where(mask_expanded == 1, weight_inside_mask, weight_outside_mask)
                            loss = loss * weight_matrix
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                
                if args.object_localization:
                    fuse_index = 5
                    localization_loss = get_object_localization_loss(
                        cross_attention_scores,
                        shadow_mask,
                        object_localization_loss_fn,
                        fuse_index,
                    )
                    loss = args.object_localization_weight * localization_loss + loss
                    clear_cross_attention_scores(cross_attention_scores)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), postfuse_module.parameters(),text_encoder_one.parameters(), text_encoder_two.parameters())
                        if args.train_text_encoder
                        else itertools.chain(unet.parameters(),postfuse_module.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        _save_accelerator_checkpoint(accelerator, args, global_step)
                accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            

            if global_step >= args.max_train_steps:
                break
            
            if args.validate_by_iter and global_step % args.validation_iterations == 0:
                if accelerator.is_main_process:
                    if _validation_enabled(args):
                        validate(args, epoch, accelerator, unet, postfuse_module, image_encoder, weight_dtype, global_step)

        if accelerator.is_main_process:
            if _validation_enabled(args) and epoch % args.validation_epochs == 0 and not args.validate_by_iter:
                validate(args, epoch, accelerator, unet, postfuse_module, image_encoder, weight_dtype, global_step)
                

    if accelerator.is_main_process:
        _save_accelerator_checkpoint(accelerator, args, global_step)

    # Save the layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
