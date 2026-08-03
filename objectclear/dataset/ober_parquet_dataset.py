import os
import io
import json
import random
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import pyarrow.parquet as pq
from tqdm import tqdm


# Mapping from the OBER parquet feature names to the internal tensor names used
# throughout training.
_PARQUET_FIELD_MAP = {
    "input": "input",
    "gt": "gt",
    "object_mask": "mask",
    "object_effect_mask": "shadow_mask",
}


def _original_id(path: str) -> str:
    """Strip the crop-index suffix from a parquet ``path`` field.

    The OBER parquet stores multiple offline crops per original image, named
    ``<orig_id>_<crop_idx>.<ext>`` (e.g. ``0427_1.jpg`` -> ``0427``). Grouping by
    the original id reproduces the legacy behaviour where every original image is
    sampled uniformly and one of its crops is picked at random per access.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" in stem:
        head, tail = stem.rsplit("_", 1)
        if tail.isdigit():
            return head
    return stem


class OBERParquetDataset(Dataset):
    """OBER dataset reading directly from HuggingFace parquet shards.

    This mirrors the legacy directory-based ``ObjectClearDataset`` exactly: it
    groups crops by their original image, exposes one entry per original image,
    and on each access randomly picks one of that image's offline crops before
    running the identical online augmentation pipeline (threshold, random
    dilation/erosion, object-centred random crop to ``input_size``, flip, colour
    augmentation, normalisation to ``[-1, 1]``).

    Args:
        parquet_dir: directory containing the ``*.parquet`` shards (the ``data/``
            folder of the downloaded dataset).
        input_size: side length of the square crop fed to the model.
        split: which split to keep (matched against the parquet ``split`` column,
            case-insensitive). ``"train"`` by default.
    """

    def __init__(
        self,
        parquet_dir,
        input_size,
        is_middle_crop=False,
        use_blank_mask=False,
        blank_mask_prob=0.4,
        color_augmentation=False,
        flip_augmentation=True,
        random_mask_dilation=True,
        random_mask_erosion=True,
        structural_mask_augment=False,
        split="train",
    ):
        self.parquet_dir = parquet_dir
        self.input_size = input_size
        self.is_middle_crop = is_middle_crop
        self.use_blank_mask = use_blank_mask
        self.blank_mask_prob = blank_mask_prob
        self.color_augmentation = color_augmentation
        self.flip_augmentation = flip_augmentation
        self.random_mask_dilation = random_mask_dilation
        self.random_mask_erosion = random_mask_erosion
        # Accepted for API compatibility with train_objectclear.py; the legacy
        # random dilation/erosion path is the only augmentation implemented, so
        # the flag is stored but does not change behaviour.
        self.structural_mask_augment = structural_mask_augment
        self.split = split.lower()

        self.shard_files = sorted(
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        )
        if not self.shard_files:
            raise FileNotFoundError(f"No .parquet shards found in {parquet_dir}")

        self._build_group_index()
        # Cache of opened ParquetFile handles, keyed by shard path (lazy, per
        # worker process).
        self._pf_cache = {}

    def _index_cache_path(self):
        return os.path.join(self.parquet_dir, f"group_index_{self.split}.json")

    def _build_group_index(self):
        """Build ``group -> [(shard_idx, row_idx), ...]`` by reading only the
        lightweight string sub-columns (never decoding image bytes)."""
        cache_path = self._index_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if cached.get("shard_files") == [os.path.basename(p) for p in self.shard_files]:
                self.group_keys = cached["group_keys"]
                # JSON turns tuples into lists; keep as lists of [shard_idx, row_idx].
                self.groups = cached["groups"]
                return

        groups = {}
        for shard_idx, shard in enumerate(
            tqdm(self.shard_files, desc="Indexing OBER parquet", unit="shard")
        ):
            table = pq.read_table(shard, columns=["input.path", "subset", "split"])
            paths = table["path"].to_pylist()
            subsets = table["subset"].to_pylist()
            splits = table["split"].to_pylist()
            for row_idx, (p, sub, sp) in enumerate(zip(paths, subsets, splits)):
                if sp is not None and sp.lower() != self.split:
                    continue
                key = f"{sub}/{_original_id(p)}"
                groups.setdefault(key, []).append([shard_idx, row_idx])

        self.group_keys = sorted(groups.keys())
        self.groups = [groups[k] for k in self.group_keys]

        with open(cache_path, "w") as f:
            json.dump(
                {
                    "shard_files": [os.path.basename(p) for p in self.shard_files],
                    "group_keys": self.group_keys,
                    "groups": self.groups,
                },
                f,
            )

    def conservative_augment_image(self, image1, image2):
        """Apply matched brightness/contrast/color/hue augmentation to a pair."""
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)

        # Brightness enhancement
        brightness_factor = random.uniform(0.8, 1.2)
        enhancer1 = ImageEnhance.Brightness(image1)
        image1 = enhancer1.enhance(brightness_factor)
        enhancer2 = ImageEnhance.Brightness(image2)
        image2 = enhancer2.enhance(brightness_factor)

        # Contrast enhancement
        contrast_factor = random.uniform(0.8, 1.2)
        enhancer1 = ImageEnhance.Contrast(image1)
        image1 = enhancer1.enhance(contrast_factor)
        enhancer2 = ImageEnhance.Contrast(image2)
        image2 = enhancer2.enhance(contrast_factor)

        # Color enhancement
        color_factor = random.uniform(0.8, 1.2)
        enhancer1 = ImageEnhance.Color(image1)
        image1 = enhancer1.enhance(color_factor)
        enhancer2 = ImageEnhance.Color(image2)
        image2 = enhancer2.enhance(color_factor)

        # Tone enhancement
        image1 = np.array(image1.convert("HSV"))
        image2 = np.array(image2.convert("HSV"))
        hue_shift = random.randint(-15, 15)
        # Compute in a wider dtype to avoid uint8 overflow on the modulo, then
        # cast back (newer numpy raises on uint8 + python int assignment).
        image1[:, :, 0] = ((image1[:, :, 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
        image2[:, :, 0] = ((image2[:, :, 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
        image1 = Image.fromarray(image1, "HSV").convert("RGB")
        image2 = Image.fromarray(image2, "HSV").convert("RGB")

        image1 = np.array(image1)
        image2 = np.array(image2)

        return image1, image2

    def random_crop_with_object_center(self, mask_image_np, input_size):
        """Pick a square crop (>= input_size) that contains the object centre."""
        H, W = mask_image_np.shape

        rotate_flag = 0
        if H > W:
            mask_image_np = cv2.rotate(mask_image_np, cv2.ROTATE_90_CLOCKWISE)
            rotate_flag = 1
            H, W = mask_image_np.shape  # update after rotation

        mask_indices = np.where(mask_image_np == 255)
        if len(mask_indices[0]) == 0 or len(mask_indices[1]) == 0:
            raise ValueError("Mask is empty, no object found.")

        center_row = int(np.mean(mask_indices[0]))
        center_col = int(np.mean(mask_indices[1]))

        max_crop_size = min(H, W)
        min_crop_size = input_size

        crop_size = random.randint(int(min_crop_size), int(max_crop_size))

        min_row = max(0, center_row - crop_size + 1)
        max_row = min(center_row, H - crop_size)
        min_col = max(0, center_col - crop_size + 1)
        max_col = min(center_col, W - crop_size)

        start_row = random.randint(min_row, max_row)
        start_col = random.randint(min_col, max_col)

        if rotate_flag == 1:
            temp = W - start_col - input_size
            start_col = start_row
            start_row = temp

        return start_row, start_col, crop_size

    def process_to_tensor(self, input_np, output_size, is_mask):
        output = input_np.astype(np.float32) / 255.0
        if is_mask:
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            output = torch.from_numpy(output).unsqueeze(0)
            output = F.interpolate(
                output.unsqueeze(0), size=(output_size, output_size), mode="nearest"
            ).squeeze(0)
        else:
            output = torch.from_numpy(output).permute(2, 0, 1).float()
            output = F.interpolate(
                output.unsqueeze(0),
                size=(output_size, output_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            output = torch.clamp(output, 0, 1)

        return output

    def __len__(self):
        return len(self.groups)

    def _get_parquet_file(self, shard_idx):
        shard = self.shard_files[shard_idx]
        pf = self._pf_cache.get(shard)
        if pf is None:
            pf = pq.ParquetFile(shard)
            self._pf_cache[shard] = pf
        return pf

    def _read_row(self, shard_idx, row_idx):
        """Decode the 4 images of a single parquet row into numpy arrays."""
        pf = self._get_parquet_file(shard_idx)
        # Locate the row group containing row_idx to avoid reading the whole shard.
        remaining = row_idx
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if remaining < n:
                table = pf.read_row_group(rg, columns=list(_PARQUET_FIELD_MAP.keys()))
                local = remaining
                break
            remaining -= n
        else:
            raise IndexError(f"row {row_idx} out of range in shard {shard_idx}")

        out = {}
        for pq_name, internal in _PARQUET_FIELD_MAP.items():
            data = table[pq_name][local].as_py()["bytes"]
            im = Image.open(io.BytesIO(data))
            if internal in ("mask", "shadow_mask"):
                out[internal] = np.array(im.convert("L"))
            else:
                out[internal] = np.array(im.convert("RGB"))
        return out

    def __getitem__(self, idx):
        # Pick one offline crop of this original image at random (legacy behaviour).
        shard_idx, row_idx = random.choice(self.groups[idx])
        row = self._read_row(shard_idx, row_idx)

        image_np = row["input"]
        gt_image_np = row["gt"]
        mask_image_np = row["mask"]
        shadow_mask_image_np = row["shadow_mask"]

        _, mask_image_np = cv2.threshold(mask_image_np, 127, 255, cv2.THRESH_BINARY)
        _, shadow_mask_image_np = cv2.threshold(
            shadow_mask_image_np, 127, 255, cv2.THRESH_BINARY
        )

        ori_mask_image_np = mask_image_np

        if self.random_mask_dilation:
            if random.random() > 0.33:
                height, width = mask_image_np.shape[:2]
                area = np.count_nonzero(mask_image_np)
                area_ratio = area / (height * width)

                if area_ratio <= 0.005:
                    kernel_size = random.randint(1, 5)
                elif area_ratio <= 0.02:
                    kernel_size = random.randint(1, 8)
                elif area_ratio <= 0.05:
                    kernel_size = random.randint(1, 12)
                elif area_ratio <= 0.1:
                    kernel_size = random.randint(1, 17)
                elif area_ratio <= 0.25:
                    kernel_size = random.randint(1, 23)
                else:
                    kernel_size = random.randint(1, 30)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_image_np = cv2.dilate(mask_image_np, kernel, iterations=1)
            elif self.random_mask_erosion:
                if random.random() > 0.5:
                    height, width = mask_image_np.shape[:2]
                    area = np.count_nonzero(mask_image_np)
                    area_ratio = area / (height * width)

                    if area_ratio <= 0.005:
                        max_k, min_k, max_attempts = 5, 1, 2
                    elif area_ratio <= 0.02:
                        max_k, min_k, max_attempts = 8, 1, 2
                    elif area_ratio <= 0.05:
                        max_k, min_k, max_attempts = 12, 1, 2
                    elif area_ratio <= 0.1:
                        max_k, min_k, max_attempts = 17, 1, 2
                    elif area_ratio <= 0.25:
                        max_k, min_k, max_attempts = 23, 1, 2
                    else:
                        max_k, min_k, max_attempts = 30, 1, 2

                    for attempt in range(max_attempts):
                        decay_ratio = attempt / (max_attempts - 1)
                        current_max = int(max_k - (max_k - min_k) * decay_ratio)
                        kernel_size = random.randint(1, current_max)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        kernel_size = max(1, kernel_size)

                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        eroded_mask = cv2.erode(mask_image_np, kernel, iterations=1)

                        if np.count_nonzero(eroded_mask) > 0:
                            mask_image_np = eroded_mask
                            break

        start_row, start_col, crop_size = self.random_crop_with_object_center(
            mask_image_np, input_size=self.input_size
        )

        if self.use_blank_mask and random.random() < self.blank_mask_prob:
            mask_image_np = np.zeros_like(mask_image_np)
            gt_image_np = image_np

        image_np = image_np[start_row:start_row + crop_size, start_col:start_col + crop_size]
        gt_image_np = gt_image_np[start_row:start_row + crop_size, start_col:start_col + crop_size]
        mask_image_np = mask_image_np[start_row:start_row + crop_size, start_col:start_col + crop_size]
        shadow_mask_image_np = shadow_mask_image_np[start_row:start_row + crop_size, start_col:start_col + crop_size]
        ori_mask_image_np = ori_mask_image_np[start_row:start_row + crop_size, start_col:start_col + crop_size]
        ori_image_np = image_np

        # Flip augmentation
        if self.flip_augmentation:
            if random.random() > 0.5:
                image_np = np.flip(image_np, axis=1)
                gt_image_np = np.flip(gt_image_np, axis=1)
                mask_image_np = np.flip(mask_image_np, axis=1)
                shadow_mask_image_np = np.flip(shadow_mask_image_np, axis=1)
                ori_mask_image_np = np.flip(ori_mask_image_np, axis=1)

        wo_color_aug_gt_np = gt_image_np
        wo_color_aug_image_np = image_np
        if self.color_augmentation:
            image_np, gt_image_np = self.conservative_augment_image(image_np, gt_image_np)

        input_tensor = self.process_to_tensor(input_np=image_np, output_size=self.input_size, is_mask=False)
        wo_color_aug_gt_tensor = self.process_to_tensor(input_np=wo_color_aug_gt_np, output_size=self.input_size, is_mask=False)
        wo_color_aug_image_tensor = self.process_to_tensor(input_np=wo_color_aug_image_np, output_size=self.input_size, is_mask=False)
        gt_tensor = self.process_to_tensor(input_np=gt_image_np, output_size=self.input_size, is_mask=False)
        mask_tensor = self.process_to_tensor(input_np=mask_image_np, output_size=self.input_size, is_mask=True)
        shadow_mask_tensor = self.process_to_tensor(input_np=shadow_mask_image_np, output_size=self.input_size, is_mask=True)
        ori_mask_tensor = self.process_to_tensor(input_np=ori_mask_image_np, output_size=self.input_size, is_mask=True)
        ori_input_tensor = self.process_to_tensor(input_np=ori_image_np, output_size=self.input_size, is_mask=False)

        input_wo_shadow = ori_mask_tensor * input_tensor + (1 - ori_mask_tensor) * gt_tensor
        obj_only_tensor = mask_tensor * input_tensor
        color_aug_obj_ori_gt_tensor = ori_mask_tensor * input_tensor + (1 - ori_mask_tensor) * wo_color_aug_gt_tensor

        outputs = {
            "input": input_tensor, "gt": gt_tensor, "mask": mask_tensor,
            "shadow_mask": shadow_mask_tensor, "ori_input_tensor": ori_input_tensor,
            "wo_color_aug_image": wo_color_aug_image_tensor, "input_wo_shadow": input_wo_shadow,
            "ori_mask_tensor": ori_mask_tensor, "obj_only_tensor": obj_only_tensor,
            "aug_obj_ori_gt": color_aug_obj_ori_gt_tensor, "prompt": "",
        }
        for k in outputs:
            if k != "prompt" and k != "mask" and k != "shadow_mask" and k != "obj_only_tensor":
                outputs[k] = outputs[k] * 2.0 - 1
        masked_image_tensor = (1 - mask_tensor) * input_tensor
        outputs["masked_image"] = masked_image_tensor
        return outputs


def save_image(image, filename):
    image_pil = TF.to_pil_image(image)
    image_pil.save(filename)
