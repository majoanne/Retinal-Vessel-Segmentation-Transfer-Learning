import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from torchvision import transforms
from PIL import Image
import pandas as pd

# ================== Configuration ==================
# Path to pretrained model (output of unet.py)
ORIGINAL_MODEL_PATH = "checkpoints/best_model.pth"

# Paths to FIVES dataset (adjust to your location)
# Download from: https://figshare.com/articles/dataset/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169
FIVES_TRAIN_IMG_DIR = "data/FIVES/train/Original"
FIVES_TRAIN_MASK_DIR = "data/FIVES/train/Ground truth"
FIVES_TEST_IMG_DIR = "data/FIVES/test/Original"
FIVES_TEST_MASK_DIR = "data/FIVES/test/Ground truth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 50
LR = 5e-5

os.makedirs("checkpoints/ablation", exist_ok=True)
os.makedirs("results/ablation", exist_ok=True)
# ====================================================

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.enc1 = self._block(n_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b); d4 = torch.cat([e4, d4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([e3, d3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([e2, d2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([e1, d1], dim=1); d1 = self.dec1(d1)
        return torch.sigmoid(self.outc(d1))

class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.7, smooth=1e-5):
        super().__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        bce_loss = nn.functional.binary_cross_entropy(pred, target)
        return self.weight * dice_loss + (1 - self.weight) * bce_loss

def calculate_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = target
    intersection = (pred_bin * target_bin).sum()
    if (pred_bin.sum() + target_bin.sum()) == 0: return 1.0
    return (2. * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-5)

def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = target
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    if union == 0: return 1.0
    return intersection / union

# Dataset for zero-shot evaluation (256x256, grayscale)
class FIVESDataset256(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        mask  = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")
        image = self.resize(image)
        mask  = self.resize(mask)

        image = np.array(image, dtype=np.float32) / 255.0
        mask  = (np.array(mask, dtype=np.float32) > 128).astype(np.float32)
        image = np.expand_dims(image, 0)
        mask  = np.expand_dims(mask, 0)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

# Dataset for fine-tuning (512x512 with preprocessing options)
class FIVESDataset512(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True, preprocess_mode='green_clahe'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.is_train = is_train
        self.preprocess_mode = preprocess_mode
        self.target_size = (512, 512)

        if is_train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.Resize(512, 512)
            ])
        else:
            self.aug = A.Compose([A.Resize(512, 512)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)

        if self.preprocess_mode == 'grayscale':
            proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.preprocess_mode == 'green':
            proc = image[:, :, 1]
        elif self.preprocess_mode == 'green_clahe':
            green = image[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            proc = clahe.apply(green)
        else:
            proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        augmented = self.aug(image=proc, mask=mask)
        image_aug = augmented['image'].astype(np.float32) / 255.0
        mask_aug = (augmented['mask'] > 128).astype(np.float32)

        return (torch.from_numpy(np.expand_dims(image_aug, 0)).float(),
                torch.from_numpy(np.expand_dims(mask_aug, 0)).float())

def evaluate(model_path, dataset_class, preprocess_mode=None, use_tta=True, desc=""):
    model = UNet(1, 1).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    if dataset_class == FIVESDataset256:
        ds = FIVESDataset256(FIVES_TEST_IMG_DIR, FIVES_TEST_MASK_DIR)
    else:
        ds = FIVESDataset512(FIVES_TEST_IMG_DIR, FIVES_TEST_MASK_DIR, is_train=False,
                             preprocess_mode=preprocess_mode)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dice_list, iou_list = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            if use_tta:
                p1 = model(images)
                p2 = torch.flip(model(torch.flip(images, [3])), [3])
                p3 = torch.flip(model(torch.flip(images, [2])), [2])
                p4 = torch.rot90(model(torch.rot90(images, 1, [2, 3])), 3, [2, 3])
                p5 = torch.rot90(model(torch.rot90(images, 3, [2, 3])), 1, [2, 3])
                outputs = (p1 + p2 + p3 + p4 + p5) / 5
            else:
                outputs = model(images)

            for i in range(images.size(0)):
                dice_list.append(calculate_dice(outputs[i:i+1], masks[i:i+1]).item())
                iou_list.append(calculate_iou(outputs[i:i+1], masks[i:i+1]).item())

    avg_dice = np.mean(dice_list)
    std_dice = np.std(dice_list)
    print(f"[{desc}] Dice: {avg_dice:.4f} ± {std_dice:.4f} | IoU: {np.mean(iou_list):.4f}")
    return avg_dice, std_dice, np.mean(iou_list)

def fine_tune(preprocess_mode):
    model_path = f"checkpoints/ablation/finetune_{preprocess_mode}.pth"
    if os.path.exists(model_path):
        return model_path

    model = UNet(1, 1).to(DEVICE)
    ckpt = torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    train_ds = FIVESDataset512(FIVES_TRAIN_IMG_DIR, FIVES_TRAIN_MASK_DIR, True, preprocess_mode)
    val_ds = FIVESDataset512(FIVES_TEST_IMG_DIR, FIVES_TEST_MASK_DIR, False, preprocess_mode)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = DiceBCELoss(weight=0.7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_dice = train_iou = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_dice += calculate_dice(outputs, masks).item()
            train_iou += calculate_iou(outputs, masks).item()

        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += calculate_dice(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks).item()

        scheduler.step()
        val_dice_avg = val_dice / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Val Dice: {val_dice_avg:.4f}")
        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save({'model_state_dict': model.state_dict(), 'best_dice': best_dice}, model_path)
    return model_path

def main():
    results = []

    d, s, i = evaluate(ORIGINAL_MODEL_PATH, FIVESDataset256, None, use_tta=False, desc="Zero-shot")
    results.append({"Variant": "Zero-shot (256 grayscale, no TTA)", "Dice": f"{d:.4f}±{s:.4f}", "IoU": i})

    modes = ['grayscale', 'green', 'green_clahe']
    for mode in modes:
        model_path = fine_tune(mode)

        d, s, i = evaluate(model_path, FIVESDataset512, mode, False, f"Fine-tune {mode} no TTA")
        results.append({"Variant": f"Fine-tune {mode} (no TTA)", "Dice": f"{d:.4f}±{s:.4f}", "IoU": i})

        d, s, i = evaluate(model_path, FIVESDataset512, mode, True, f"Fine-tune {mode} + 5-way TTA")
        results.append({"Variant": f"Fine-tune {mode} + 5-way TTA", "Dice": f"{d:.4f}±{s:.4f}", "IoU": i})

    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print(df.to_string(index=False))
    print("="*100)
    df.to_csv("results/ablation/ablation_results.csv", index=False)

if __name__ == "__main__":
    main()