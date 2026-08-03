import os
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import json

class ObjectClearDataset(Dataset):
    def __init__(self, image_dir, input_size, is_middle_crop=False, use_blank_mask=False, blank_mask_prob=0.4, color_augmentation=False, flip_augmentation=True, random_mask_dilation=True, random_mask_erosion=True, structural_mask_augment=False):
        self.image_dir = image_dir
        self.input_size = input_size
        self.is_middle_crop = is_middle_crop
        self.use_blank_mask = use_blank_mask
        self.blank_mask_prob = blank_mask_prob
        self.color_augmentation = color_augmentation
        self.flip_augmentation  = flip_augmentation
        self.random_mask_dilation = random_mask_dilation
        self.random_mask_erosion = random_mask_erosion
        # Accepted for API compatibility with train_objectclear.py. This dataset
        # version only implements the legacy random dilation/erosion path, so the
        # flag is stored but does not change behaviour.
        self.structural_mask_augment = structural_mask_augment
        
        self.paths_file = os.path.join(image_dir, 'paths_file.json')
        
        if os.path.exists(self.paths_file):
            with open(self.paths_file, 'r') as f:
                paths_info = json.load(f)
                self.image_paths = paths_info['image_paths']
                self.gt_paths = paths_info['gt_paths']
                self.mask_paths = paths_info['mask_paths']
                self.dir_paths = paths_info['dir_paths']
                self.shadow_mask_paths = paths_info['shadow_mask_paths']
        else:
            self.image_paths = []
            self.gt_paths = []
            self.mask_paths = []
            self.dir_paths = []
            self.shadow_mask_paths = []

            for subdir, _, files in tqdm(os.walk(image_dir), desc="Processing directories", unit="dir"):
                if any(file.startswith('input_') for file in files):
                    self.dir_paths.append(subdir)
                for file in files:
                    if file.startswith('input_'):
                        result_path = os.path.join(subdir, file)
                        gt_path = os.path.join(subdir, file.replace('input_', 'gt_'))
                        mask_path = os.path.join(subdir, file.replace('input_', 'mask_').replace('jpg', 'png'))
                        shadow_mask_path = os.path.join(subdir, file.replace('input_', 'shadow_mask_').replace('jpg', 'png'))
                        self.image_paths.append(result_path)
                        if os.path.exists(gt_path):
                            self.gt_paths.append(gt_path)
                        if os.path.exists(mask_path):
                            self.mask_paths.append(mask_path)
                        if os.path.exists(shadow_mask_path):
                            self.shadow_mask_paths.append(shadow_mask_path)
                paths_info = {
                    'image_paths': self.image_paths,
                    'gt_paths': self.gt_paths,
                    'mask_paths': self.mask_paths,
                    'dir_paths': self.dir_paths,
                    'shadow_mask_paths': self.shadow_mask_paths
                }
                with open(self.paths_file, 'w') as f:
                    json.dump(paths_info, f)
        
        self.image_paths = sorted(self.image_paths)
        self.gt_paths = sorted(self.gt_paths)
        self.mask_paths = sorted(self.mask_paths)
        self.shadow_mask_paths = sorted(self.shadow_mask_paths)
        self.dir_paths = sorted(self.dir_paths)
        
        assert len(self.image_paths) == len(self.mask_paths), f"Lengths of image_paths and mask_paths must be equal"
        assert len(self.image_paths) == len(self.shadow_mask_paths), f"Lengths of image_paths and shdow_mask_paths must be equal"
        
    def conservative_augment_image(self, image1, image2):
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
        image1 = np.array(image1.convert('HSV'))
        image2 = np.array(image2.convert('HSV'))
        hue_shift = random.randint(-15, 15)
        image1[:, :, 0] = ((image1[:, :, 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
        image2[:, :, 0] = ((image2[:, :, 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
        image1 = Image.fromarray(image1, 'HSV').convert('RGB')
        image2 = Image.fromarray(image2, 'HSV').convert('RGB')
        
        image1 = np.array(image1)
        image2 = np.array(image2)

        return image1, image2
                    
    def crop_image_by_mask(self, mask_image_np, input_size, is_full_crop=False):
        H, W = mask_image_np.shape
        
        rotate_flag = 0
        if H>W:
            mask_image_np = cv2.rotate(mask_image_np, cv2.ROTATE_90_CLOCKWISE)
            rotate_flag = 1
            
        H, W = mask_image_np.shape
        
        margin_ratio = 0.1
        mask_indices = np.where(mask_image_np == 255)
        min_row, max_row = mask_indices[0].min(), mask_indices[0].max()
        min_col, max_col = mask_indices[1].min(), mask_indices[1].max()
        
        if is_full_crop: # Full crop will resize the obj if it is too long.
            min_crop_size = max(H, min(W, int(1.1*(max_col-min_col))))
            if 1.1*(max_col - min_col) > H:
                max_crop_size = W
            else:
                max_crop_size = H
            crop_size = random.randint(min_crop_size, max_crop_size+1)
            start_row = 0
            start_col = random.randint(max(0, min(max_col, W)-crop_size), min(W-crop_size, max(min_col, 0))+1)
        else:
            if 1.1*(max_col - min_col) > H:
                max_crop_size = W
                min_crop_size = min(W, int(1.11*(max_col-min_col)))
            else:
                max_crop_size = H
                min_crop_size = max(input_size, min(H, int(1.11*(max_row-min_row))), min(H, int(1.11*(max_col-min_col))))
                
            crop_size = random.randint(min_crop_size, max_crop_size+1)
            margin_size = int(crop_size*margin_ratio)
            start_row = random.randint(max(0, min(max_row+margin_size, H)-crop_size), min(max(H-crop_size, 0), max(min_row, 0))+1)
            start_col = random.randint(max(0, min(max_col, W)-crop_size), min(W-crop_size, max(min_col, 0))+1)
            
        if rotate_flag==1:
            temp = W - start_col - input_size
            start_col = start_row
            start_row = temp
            
    
        return start_row, start_col, crop_size
    

    def random_crop_with_object_center(self, mask_image_np, input_size):
        H, W = mask_image_np.shape

        rotate_flag = 0
        if H > W:
            mask_image_np = cv2.rotate(mask_image_np, cv2.ROTATE_90_CLOCKWISE)
            rotate_flag = 1
            H, W = mask_image_np.shape  # update after rotation

        # Find the non-zero (object) region of the mask.
        mask_indices = np.where(mask_image_np == 255)
        if len(mask_indices[0]) == 0 or len(mask_indices[1]) == 0:
            raise ValueError("Mask is empty, no object found.")

        # Compute the mask centre.
        center_row = int(np.mean(mask_indices[0]))
        center_col = int(np.mean(mask_indices[1]))

        max_crop_size = min(H, W)
        min_crop_size = input_size

        crop_size = random.randint(int(min_crop_size), int(max_crop_size))

        # Pick a random square crop that contains the object centre.
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
            output = F.interpolate(output.unsqueeze(0), size=(output_size, output_size), mode='nearest').squeeze(0)
        else:
            output = torch.from_numpy(output).permute(2, 0, 1).float()
            output = F.interpolate(output.unsqueeze(0), size=(output_size, output_size), mode='bicubic', align_corners=False).squeeze(0)
            output = torch.clamp(output, 0, 1)

        return output


    def __len__(self):
        return len(self.dir_paths)

    def __getitem__(self, idx):
        dir_path = self.dir_paths[idx]
        matching_indices = [i for i, path in enumerate(self.image_paths) if path.startswith(dir_path)]
        selected_index = random.choice(matching_indices)
        image_path = self.image_paths[selected_index]
        gt_path = self.gt_paths[selected_index]
        mask_path = self.mask_paths[selected_index]
        shadow_mask_path = self.shadow_mask_paths[selected_index]
        
        is_full_crop = "_full" in image_path
        
        img_name = image_path.split('/')[-2]
        assert img_name==gt_path.split('/')[-2], mask_path.split('/')[-2]
        assert img_name==gt_path.split('/')[-2], shadow_mask_path.split('/')[-2]
        
        image_np = np.array(Image.open(image_path).convert('RGB'))
        gt_image_np = np.array(Image.open(gt_path).convert('RGB'))
        mask_image_np = np.array(Image.open(mask_path).convert('L'))
        _, mask_image_np = cv2.threshold(mask_image_np, 127, 255, cv2.THRESH_BINARY)
        shadow_mask_image_np = np.array(Image.open(shadow_mask_path).convert('L'))
        _, shadow_mask_image_np = cv2.threshold(shadow_mask_image_np, 127, 255, cv2.THRESH_BINARY)

        
        ori_mask_image_np = mask_image_np
        
        if self.random_mask_dilation:
            if random.random() > 0.33:
                height, width = mask_image_np.shape[:2]
                area = np.count_nonzero(mask_image_np)
                area_ratio = area / (height * width)

                # Set the dilation kernel size dynamically based on object area.
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

                    # Set parameters dynamically based on object area.
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

                    # Try eroding several times, shrinking the kernel size smoothly.
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
            
        start_row, start_col, crop_size = self.random_crop_with_object_center(mask_image_np, input_size=self.input_size)
        
        if self.use_blank_mask and random.random() < self.blank_mask_prob:
            mask_image_np = np.zeros_like(mask_image_np)
            gt_image_np = image_np
            
        image_np = image_np[start_row:start_row+crop_size, start_col:start_col+crop_size]
        gt_image_np = gt_image_np[start_row:start_row+crop_size, start_col:start_col+crop_size]
        mask_image_np = mask_image_np[start_row:start_row+crop_size, start_col:start_col+crop_size]
        shadow_mask_image_np = shadow_mask_image_np[start_row:start_row+crop_size, start_col:start_col+crop_size]
        ori_mask_image_np = ori_mask_image_np[start_row:start_row+crop_size, start_col:start_col+crop_size]
        ori_image_np = image_np
        
        # Flip augmentation
        if self.flip_augmentation:
            if random.random() > 0.5:
                image_np = np.flip(image_np, axis=1)  # Horizontal flip
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

        outputs = {'input': input_tensor, 'gt': gt_tensor, 'mask': mask_tensor, 'shadow_mask': shadow_mask_tensor, 'ori_input_tensor': ori_input_tensor, 'wo_color_aug_image': wo_color_aug_image_tensor, \
                   'input_wo_shadow': input_wo_shadow, 'ori_mask_tensor': ori_mask_tensor, 'obj_only_tensor': obj_only_tensor, 'aug_obj_ori_gt':color_aug_obj_ori_gt_tensor, 'prompt': ''}
        for k in outputs:
            if k!='prompt' and k!='mask' and k!='shadow_mask' and k!='obj_only_tensor':
                outputs[k] = outputs[k] * 2.0 - 1
        masked_image_tensor = (1 - mask_tensor) * input_tensor
        outputs['masked_image'] = masked_image_tensor
        return outputs
