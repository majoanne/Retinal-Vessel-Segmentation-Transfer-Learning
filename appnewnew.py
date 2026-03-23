import os
import tempfile
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import zipfile
import io
from pathlib import Path

BASE_DIR = Path(".")
MODEL_DIR = BASE_DIR / "models"
TEST_IMAGES_PNG_DIR = BASE_DIR / "test_images_png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.enc1 = self._block(n_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = self._block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._block(64, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([e4, self.up4(b)], 1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], 1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], 1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], 1))
        return torch.sigmoid(self.outc(d1))

def load_model():
    model_path = MODEL_DIR / "finetune_green_clahe.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = UNet().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

model = load_model()

def apply_clahe(image_np, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(image_np)

def preprocess_for_model(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        green = image[:, :, 1]
    else:
        green = image
    proc = apply_clahe(green)
    proc = cv2.resize(proc, (512, 512))
    tensor = torch.from_numpy(proc.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor

def predict(image):
    tensor = preprocess_for_model(image)
    with torch.no_grad():
        pred = model(tensor)
        mask = (pred > 0.5).float().cpu().squeeze().numpy()
    return mask

def load_tiff_for_preview(file):
    if file is None:
        return None
    img = Image.open(file.name)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def single_predict(image):
    if image is None:
        raise gr.Error("Please upload an image")
    h, w = image.shape[:2]
    mask_256 = predict(image)
    mask_orig = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_uint8 = (mask_orig > 0.5).astype(np.uint8) * 255
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[mask_orig > 0.5] = [255, 0, 0]
    overlay = overlay.astype(np.uint8)
    return np.ascontiguousarray(mask_uint8), np.ascontiguousarray(overlay)

def batch_predict(files):
    if files is None or len(files) == 0:
        raise gr.Error("Please upload at least one image")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
        zip_path = tmp_zip.name

    try:
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in files:
                try:
                    if hasattr(file, 'name'):
                        file_path = file.name
                    elif isinstance(file, str):
                        file_path = file
                    else:
                        raise TypeError(f"Unsupported file type: {type(file)}")

                    img = Image.open(file_path).convert('RGB')
                    orig = np.array(img)
                    h, w = orig.shape[:2]

                    mask_256 = predict(orig)
                    mask_orig = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask_orig > 0.5).astype(np.uint8) * 255

                    mask_img = Image.fromarray(mask_bin)
                    img_buffer = io.BytesIO()
                    mask_img.save(img_buffer, format='PNG')

                    base_name = os.path.basename(file_path)
                    out_name = f"{os.path.splitext(base_name)[0]}_mask.png"
                    zf.writestr(out_name, img_buffer.getvalue())

                except Exception as e:
                    import traceback
                    error_msg = f"Error processing {getattr(file, 'name', str(file))}:\n{traceback.format_exc()}"
                    print(error_msg)
                    raise gr.Error(f"Batch processing failed: {str(e)}")

        return zip_path

    except Exception as e:
        if os.path.exists(zip_path):
            os.unlink(zip_path)
        raise e

def generate_mask_and_overlay(image):
    if image is None:
        return None, None, None
    h, w = image.shape[:2]
    mask_256 = predict(image)
    mask_orig = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_uint8 = (mask_orig > 0.5).astype(np.uint8) * 255
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[mask_orig > 0.5] = [255, 0, 0]
    overlay = overlay.astype(np.uint8)
    overlay_resized = cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_LINEAR)
    return (np.ascontiguousarray(mask_uint8),
            np.ascontiguousarray(overlay),
            np.ascontiguousarray(overlay_resized))

def download_final_mask(image, ai_mask, editor_output):
    try:
        if image is None:
            raise gr.Error("Please upload an original image")
        if ai_mask is None:
            raise gr.Error("Please generate an AI mask first")
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        h, w = gray.shape
        if ai_mask.dtype != np.uint8:
            ai_mask = ai_mask.astype(np.uint8)
        if ai_mask.shape[:2] != (h, w):
            ai_mask = cv2.resize(ai_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (ai_mask > 128).astype(np.uint8) * 255
        if editor_output is not None and isinstance(editor_output, dict):
            layers = editor_output.get('layers', [])
            if layers and len(layers) > 0:
                layer = layers[0]
                if layer is not None and layer.shape[-1] == 4:
                    if layer.shape[:2] != (h, w):
                        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_NEAREST)
                    alpha = layer[:, :, 3] > 0
                    r = layer[:, :, 0]
                    add = alpha & (r > 127)
                    erase = alpha & (r <= 127)
                    mask_bin[add] = 255
                    mask_bin[erase] = 0
        mask_img = Image.fromarray(mask_bin)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            mask_img.save(tmp.name)
            return tmp.name
    except Exception as e:
        raise gr.Error(f"Download failed: {str(e)}")

algorithm_md = """# 🔬 Algorithm Principles & Results

## U-Net Architecture
U-Net is a symmetric encoder-decoder network designed for biomedical image segmentation (Ronneberger et al., 2015).  
**Key design**:
- **Contracting path (encoder)**: successive convolutions + pooling extract multi-scale features
- **Expanding path (decoder)**: transposed convolutions upsample to recover resolution
- **Skip connections**: fuse high- and low-level features to preserve vessel edge details

## Enhancement Techniques
- **Multi-dataset transfer learning**: pre-trained on **STARE, DRIVE, CHASE_DB1, HRF** datasets, achieving internal validation Dice of **0.9824**.
- **FIVES fine-tuning**: fine-tuned on the external FIVES dataset using **green channel + CLAHE** preprocessing, achieving Dice = **0.8971 ± 0.1022** on test set.

## 📈 Performance Highlights
- Internal validation (four public datasets) best Dice = **0.9824**
- FIVES external test (green+CLAHE) Dice = **0.8971 ± 0.1022**
"""

explore_md = """## 🛠️ Explore More: Resources for You
**Want to experiment yourself? Want to learn more? Everything is neatly organized here — one-click access!**

### 📥 Datasets (Download original images + annotations directly — try our AI or train your own)
- **FIVES** (800 high-resolution multi-disease images — test set used in this app) → [One-click download page](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)
- **DRIVE** → [Official download](https://drive.grand-challenge.org/)
- **STARE** → [Official download](https://cecas.clemson.edu/~ahoover/stare/)
- **CHASE_DB1** → [Official download](https://datasetninja.com/chase-db1)
- **HRF** → [Official download](https://www5.cs.fau.de/research/data/fundus-images/)

### 💻 Open-Source Projects (Free code — run directly or customize into your own tool)
- [berenslab/Retinal-Vessel-Segmentation-Benchmark](https://github.com/berenslab/Retinal-Vessel-Segmentation-Benchmark) — Multi-model benchmark with FIVES support
- [roisantos/vesselview](https://github.com/roisantos/vesselview) — Compact U-Net optimized for FIVES
- [agaldran/lwnet](https://github.com/agaldran/lwnet) — Ultra-lightweight model (70K parameters)
- [lee-zq/VesselSeg-Pytorch](https://github.com/lee-zq/VesselSeg-Pytorch) — Complete PyTorch training & testing toolkit
- [kahnertk/retinal-vessel-segmentation](https://github.com/kahnertk/retinal-vessel-segmentation) — Gradio web interface (similar to this app, deployable)

**Browse all projects**: https://github.com/topics/retinal-vessel-segmentation

### 📚 Quick Learning Resources (Beginner-friendly)
- YouTube: [U-Net for Retina Blood Vessel Segmentation in PyTorch](https://www.youtube.com/watch?v=T0BiFBaMLDQ)
- Kaggle Notebook: [Retinal Vessel Segmentation examples](https://www.kaggle.com/search?q=retinal+vessel+segmentation)

**After exploring these resources, feel free to come back and use our app for doctor-level manual refinement!**
"""

css = """
.tab-nav button,
[role="tablist"] button,
button[data-testid="tab-button"],
.tab-nav > button,
.tabs > .tab-nav button,
.tab-buttons button {
    font-size: 18px !important;
    font-weight: 600 !important;
    line-height: 1.3 !important;
}

/* Make images in rows with class "square-row" square and fill the container */
.square-row .gr-image {
    aspect-ratio: 1 / 1;
    object-fit: cover;
    width: 100%;
    height: auto;
}
"""

with gr.Blocks(title="Retinal Vessel AI Annotator", css=css) as demo:
    with gr.Tab("📚 Algorithm & Background"):
        gr.Markdown(algorithm_md)
        gr.Image(str(BASE_DIR / "unet.png"), label="U-Net architecture diagram (from original paper)", width=600)

    with gr.Tab("📸 Single Image Annotation"):
        with gr.Row(elem_classes="square-row"):
            with gr.Column(scale=1):
                input_img = gr.Image(label="Input Image (jpg/png)", type="numpy")
            with gr.Column(scale=1):
                output_mask = gr.Image(label="AI Segmentation Mask", type="numpy")
            with gr.Column(scale=1):
                output_overlay = gr.Image(label="Vessel Overlay (red)", type="numpy")

        with gr.Row():
            with gr.Column(scale=1):
                tif_upload_single = gr.UploadButton("📁 Upload TIF file", file_types=[".tif", ".tiff"])
            with gr.Column(scale=2):
                run_single = gr.Button("Run Segmentation", variant="primary")

        example_pngs = [
            [str(TEST_IMAGES_PNG_DIR / "33_A.png")],
            [str(TEST_IMAGES_PNG_DIR / "505_N.png")]
        ]
        gr.Examples(
            examples=example_pngs,
            inputs=[input_img],
            label="Test image examples (click to load)"
        )

        tif_upload_single.upload(fn=load_tiff_for_preview, inputs=tif_upload_single, outputs=input_img)
        run_single.click(fn=single_predict, inputs=[input_img], outputs=[output_mask, output_overlay])

    with gr.Tab("📂 Batch Processing"):
        files = gr.File(label="Upload multiple images", file_count="multiple")
        run_batch = gr.Button("Run Batch Segmentation", variant="primary")
        result_zip = gr.File(label="Download all masks (ZIP)")
        run_batch.click(fn=batch_predict, inputs=[files], outputs=result_zip)

    with gr.Tab("🖌️ Manual Refinement"):
        with gr.Row(elem_classes="square-row"):
            with gr.Column(scale=1):
                img_correct = gr.Image(label="Original Image (jpg/png)", type="numpy")
            with gr.Column(scale=1):
                ai_mask_disp = gr.Image(label="AI Mask", type="numpy", interactive=False)
            with gr.Column(scale=1):
                overlay_disp = gr.Image(label="Vessel Overlay (red)", type="numpy", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                tif_upload_correct = gr.UploadButton("📁 Upload TIF file", file_types=[".tif", ".tiff"])
            with gr.Column(scale=2):
                gen_mask_btn = gr.Button("Generate AI Mask", variant="primary")

        gr.Examples(
            examples=example_pngs,
            inputs=[img_correct],
            label="Test image examples (click to load)"
        )

        gr.Markdown("---")
        gr.Markdown("### Edit corrections: draw white to add vessels, black to erase")

        editor = gr.ImageEditor(
            label="Draw corrections",
            type="numpy",
            brush=gr.Brush(colors=["#FFFFFF", "#000000"]),
            eraser=True,
            layers=True,
            height=600
        )

        download_btn = gr.DownloadButton("💾 Download Final Corrected Mask", variant="primary")

        tif_upload_correct.upload(fn=load_tiff_for_preview, inputs=tif_upload_correct, outputs=img_correct)
        gen_mask_btn.click(fn=generate_mask_and_overlay, inputs=[img_correct], outputs=[ai_mask_disp, overlay_disp, editor])
        download_btn.click(
            fn=download_final_mask,
            inputs=[img_correct, ai_mask_disp, editor],
            outputs=download_btn
        )

    with gr.Tab("🛠️ Explore More"):
        gr.Markdown(explore_md)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(TEST_IMAGES_PNG_DIR)],
        theme=gr.themes.Soft()
    )