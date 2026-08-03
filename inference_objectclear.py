import os
import argparse
import glob
import torch
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side
from PIL import Image



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/imgs', 
                        help='Input image or folder. Default: inputs/imgs')
    parser.add_argument('-m', '--mask_path', type=str, default='./inputs/masks',
                        help='Input mask image or folder. Default: inputs/masks')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                        help='Output folder. Default: results/<input_name>')
    parser.add_argument('--cache_dir', type=str,
                        default=os.environ.get('HF_HOME', None),
                        help="Path to cache directory")
    parser.add_argument('--use_fp16', action='store_true', 
                        help='Use float16 for inference')
    parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for torch.Generator. Default: 42')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of diffusion inference steps. Default: 20')
    parser.add_argument('--guidance_scale', type=float, default=2.5, 
                        help='CFG guidance scale. Default: 2.5')
    parser.add_argument('--no_agf', action='store_true', 
                        help='Disable Attention Guided Fusion')
    args = parser.parse_args()
    
    
    # ------------------------ input & output ------------------------
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_single_img'
    else: # input img folder
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'
        
    if args.mask_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single mask path
        input_mask_list = [args.mask_path]
    else: # input mask folder
        # scan all the jpg and png masks
        input_mask_list = sorted(glob.glob(os.path.join(args.mask_path, '*.[jpJP][pnPN]*[gG]')))
        
    if len(input_img_list) != len(input_mask_list):
        raise ValueError(f"Mismatch between input images ({len(input_img_list)}) and masks ({len(input_mask_list)}).")

    if not args.output_path is None: # set output path
        result_root = args.output_path
        
    os.makedirs(result_root, exist_ok=True)

    test_img_num = len(input_img_list)
    
    
    # ------------------ set up ObjectClear pipeline -------------------
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    variant = "fp16" if args.use_fp16 else None
    use_agf = not args.no_agf
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=use_agf,
        cache_dir=args.cache_dir,
        variant=variant,
    )
    pipe.to(device)
    
    
    # -------------------- start to processing ---------------------
    for i, (img_path, mask_path) in enumerate(zip(input_img_list, input_mask_list)):
        img_name = os.path.basename(img_path)
        basename, _ = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_or = image.copy()
        
        # Our model was trained on 512×512 resolution.
        # Resizing the input so that the **shorter side is 512** helps achieve the best performance.
        image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
        mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        
        w, h = image.size

        generator = torch.Generator(device=device).manual_seed(args.seed)

        result = pipe(
            prompt="remove the instance of object",
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )
        
        fused_img_pil = result.images[0]

        # save results
        save_path = os.path.join(result_root, f'{basename}.png')
        fused_img_pil = fused_img_pil.resize(image_or.size)
        fused_img_pil.save(save_path)

    print(f'\nAll results are saved in {result_root}')