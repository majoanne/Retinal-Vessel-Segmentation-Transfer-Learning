import os
import shutil
import random
import cv2
from PIL import Image
from torchvision import transforms
import albumentations as A

# Supported image formats
SUPPORTED_FORMATS = {
    'image': ['.ppm', '.jpg', '.jpeg', '.tif', '.tiff', '.png', '.gif'],
    'mask': ['.ppm', '.jpg', '.jpeg', '.tif', '.tiff', '.png', '.gif']
}

# Data directory configuration (modify according to your dataset location)
DATA_CONFIG = {
    'new_data': {
        'image_dir': "data/yuantu",      # Original image folder
        'mask_dir': "data/biaoji",       # Ground truth mask folder
        'mask_suffix': ''
    }
}

# Processing output configuration
PROCESS_CONFIG = {
    'output_png_dir': "output/png_unified",          # PNG converted output
    'output_resized_dir': "output/resized",          # Resized images output
    'target_data_dir': "data/unet_segmentation",     # Final dataset directory
    'target_size': (256, 256),
    'train_ratio': 0.8
}

def get_file_prefix(filename):
    return os.path.splitext(filename)[0]

def collect_files():
    all_images = {}
    all_masks = {}

    for dataset_name, config in DATA_CONFIG.items():
        # Collect images
        for filename in os.listdir(config['image_dir']):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_FORMATS['image']:
                prefix = get_file_prefix(filename)
                all_images[prefix] = os.path.join(config['image_dir'], filename)

        # Collect masks
        for filename in os.listdir(config['mask_dir']):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_FORMATS['mask']:
                prefix = get_file_prefix(filename)
                all_masks[prefix] = os.path.join(config['mask_dir'], filename)

    # Match images and masks with the same prefix
    matched_pairs = []
    common_prefixes = set(all_images.keys()) & set(all_masks.keys())
    for prefix in common_prefixes:
        matched_pairs.append({
            'prefix': prefix,
            'image': all_images[prefix],
            'mask': all_masks[prefix],
        })
    return matched_pairs

def main():
    matched_pairs = collect_files()
    if not matched_pairs:
        print("No matching image-mask pairs found. Please check the data directories.")
        return

    # Step 1: Convert all images and masks to PNG (grayscale)
    os.makedirs(PROCESS_CONFIG['output_png_dir'], exist_ok=True)
    img_png_dir = os.path.join(PROCESS_CONFIG['output_png_dir'], "images")
    mask_png_dir = os.path.join(PROCESS_CONFIG['output_png_dir'], "masks")
    os.makedirs(img_png_dir, exist_ok=True)
    os.makedirs(mask_png_dir, exist_ok=True)

    convert_count = 0
    for pair in matched_pairs:
        try:
            img = Image.open(pair['image']).convert("L")
            img.save(os.path.join(img_png_dir, f"{pair['prefix']}.png"))

            mask = Image.open(pair['mask']).convert("L")
            mask.save(os.path.join(mask_png_dir, f"{pair['prefix']}.png"))

            convert_count += 1
        except Exception as e:
            print(f"Warning: Failed to convert {pair['prefix']}: {e}")
    print(f"Step 1 completed: {convert_count} image-mask pairs converted to PNG")

    if convert_count == 0:
        return

    # Step 2: Resize all images and masks to target size
    os.makedirs(PROCESS_CONFIG['output_resized_dir'], exist_ok=True)
    img_resize_dir = os.path.join(PROCESS_CONFIG['output_resized_dir'], "images")
    mask_resize_dir = os.path.join(PROCESS_CONFIG['output_resized_dir'], "masks")
    os.makedirs(img_resize_dir, exist_ok=True)
    os.makedirs(mask_resize_dir, exist_ok=True)

    resize = transforms.Resize(PROCESS_CONFIG['target_size'])
    resize_count = 0
    for filename in os.listdir(img_png_dir):
        if not filename.endswith(".png"):
            continue
        img_path = os.path.join(img_png_dir, filename)
        mask_path = os.path.join(mask_png_dir, filename)
        if not os.path.exists(mask_path):
            continue
        try:
            img = Image.open(img_path).convert("L")
            resized_img = resize(img)
            mask = Image.open(mask_path).convert("L")
            resized_mask = resize(mask)

            resized_img.save(os.path.join(img_resize_dir, filename))
            resized_mask.save(os.path.join(mask_resize_dir, filename))
            resize_count += 1
        except Exception as e:
            print(f"Warning: Failed to resize {filename}: {e}")
    print(f"Step 2 completed: {resize_count} image-mask pairs resized to {PROCESS_CONFIG['target_size']}")

    if resize_count == 0:
        return

    # Step 3: Split dataset into training and validation sets
    train_img_dir = os.path.join(PROCESS_CONFIG['target_data_dir'], "train/images")
    train_mask_dir = os.path.join(PROCESS_CONFIG['target_data_dir'], "train/masks")
    val_img_dir = os.path.join(PROCESS_CONFIG['target_data_dir'], "val/images")
    val_mask_dir = os.path.join(PROCESS_CONFIG['target_data_dir'], "val/masks")
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    image_names = [f for f in os.listdir(img_resize_dir) if f.endswith(".png")]
    random.shuffle(image_names)
    train_size = int(PROCESS_CONFIG['train_ratio'] * len(image_names))
    train_names = image_names[:train_size]
    val_names = image_names[train_size:]

    for name in train_names:
        shutil.copy(os.path.join(img_resize_dir, name), train_img_dir)
        shutil.copy(os.path.join(mask_resize_dir, name), train_mask_dir)

    for name in val_names:
        shutil.copy(os.path.join(img_resize_dir, name), val_img_dir)
        shutil.copy(os.path.join(mask_resize_dir, name), val_mask_dir)

    print(f"Step 3 completed: Training set = {len(train_names)} images, Validation set = {len(val_names)} images")

    # Step 4: Data augmentation on training set (optional)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

    aug_count = 0
    for filename in os.listdir(train_img_dir):
        if not filename.endswith(".png") or "_aug" in filename:
            continue
        img_path = os.path.join(train_img_dir, filename)
        mask_path = os.path.join(train_mask_dir, filename)
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        if img is None or mask is None:
            continue
        try:
            for i in range(2):
                augmented = transform(image=img, mask=mask)
                aug_img = augmented["image"]
                aug_mask = augmented["mask"]
                base_name = os.path.splitext(filename)[0]
                cv2.imwrite(os.path.join(train_img_dir, f"{base_name}_aug{i}.png"), aug_img)
                cv2.imwrite(os.path.join(train_mask_dir, f"{base_name}_aug{i}.png"), aug_mask)
                aug_count += 1
        except Exception as e:
            print(f"Warning: Failed to augment {filename}: {e}")
    print(f"Step 4 completed: Generated {aug_count} augmented image-mask pairs")

    print("All preprocessing steps completed successfully.")

if __name__ == "__main__":
    main()