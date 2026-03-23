Retinal Vessel Segmentation with Transfer Learning
Official code for the paper "Enhancing Generalization in Retinal Vessel Segmentation: A Transfer Learning Framework with Robust Domain Calibration and Clinical Web Integration".
This repository contains the code for retinal vessel segmentation research. The code includes data preprocessing, multi-source pre-training on STARE/DRIVE/CHASE_DB1/HRF, fine-tuning/ablation studies on the FIVES dataset, and a clinical web demo.

Project Structure
processing.py - Data preprocessing code
unet.py - U-Net model and multi-source pre-training code
ablation.py - FIVES fine-tuning and ablation study code
appnewnew.py - Web demo for clinical retinal vessel segmentation visualization
requirement.txt - Python environment dependencies

Quick Start
Set up the environment:
conda create -n retinal-seg python=3.8
conda activate retinal-seg
pip install -r requirement.txt

Prepare the datasets:
Download STARE, DRIVE, CHASE_DB1, HRF, and FIVES from their official websites.
Place the raw images and annotations in the correct folders (paths can be modified in the code).

Run the code:
python processing.py
python unet.py
python ablation.py
# Run the web demo (requires models/ folder for weights and test_images_png/ for sample images)
python appnewnew.py

Key Results
The best performance on FIVES test set is 0.8971 ± 0.1022 Dice coefficient (using green channel extraction + CLAHE, no TTA).

Citation
If you use this code, please cite our paper:
@article{yourname2026retinal,
title={Enhancing Generalization in Retinal Vessel Segmentation: A Transfer Learning Framework with Robust Domain Calibration},
author={Your Name},
journal={Your Journal},
year={2026}
}

License
This project is licensed under the MIT License.