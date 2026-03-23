import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# ================== Configuration ==================
data_dir = "data/unet_segmentation"                     # Relative path to dataset
train_img_dir = os.path.join(data_dir, "train/images")
train_mask_dir = os.path.join(data_dir, "train/masks")
val_img_dir = os.path.join(data_dir, "val/images")
val_mask_dir = os.path.join(data_dir, "val/masks")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
IN_CHANNELS = 1
OUT_CHANNELS = 1

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
# ====================================================

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 128).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

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

        d4 = self.up4(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.outc(d1))

class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1e-5):
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
    if (pred_bin.sum() + target_bin.sum()) == 0:
        return 1.0
    return (2. * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-5)

def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = target
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dice += calculate_dice(outputs, masks).item()
        total_iou += calculate_iou(outputs, masks).item()

    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_dice += calculate_dice(outputs, masks).item()
            total_iou += calculate_iou(outputs, masks).item()
    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

class LRScheduler:
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr != old_lr:
                param_group['lr'] = new_lr

def save_training_curves(train_losses, train_dices, train_ious,
                         val_losses, val_dices, val_ious):
    try:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(train_dices, label='Train Dice', color='blue')
        plt.plot(val_dices, label='Val Dice', color='red')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(train_ious, label='Train IoU', color='blue')
        plt.plot(val_ious, label='Val IoU', color='red')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

def visualize_predictions_simple(model, val_loader, device):
    try:
        model.eval()
        images, masks = next(iter(val_loader))
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)

        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        for i in range(min(3, len(images))):
            img_np = images[i].cpu().squeeze().numpy()
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1} - Input')
            axes[i, 0].axis('off')

            mask_np = masks[i].cpu().squeeze().numpy()
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            pred_np = outputs[i].cpu().squeeze().numpy()
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

            pred_bin = (pred_np > 0.5).astype(np.float32)
            axes[i, 3].imshow(pred_bin, cmap='gray')
            axes[i, 3].set_title('Binary Prediction')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig('results/predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

def main():
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir)
    val_dataset = SegmentationDataset(val_img_dir, val_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNet(IN_CHANNELS, OUT_CHANNELS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE}, Total parameters: {total_params:,}")

    criterion = DiceBCELoss(weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = LRScheduler(optimizer, patience=10, factor=0.5)

    train_losses, train_dices, train_ious = [], [], []
    val_losses, val_dices, val_ious = [], [], []
    best_dice = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = val_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_dices.append(train_dice)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
            print(f"-> Best model saved (Dice: {best_dice:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            print(f"-> Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    print(f"Best validation Dice: {best_dice:.4f}")

    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    save_training_curves(train_losses, train_dices, train_ious, val_losses, val_dices, val_ious)
    visualize_predictions_simple(model, val_loader, DEVICE)

if __name__ == "__main__":
    main()