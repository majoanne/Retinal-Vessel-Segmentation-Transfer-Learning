Retinal Vessel Segmentation with Transfer Learning
Official code for the paper "Enhancing Generalization in Retinal Vessel Segmentation: A Transfer Learning Framework with Robust Domain Calibration and Clinical Web Integration".
This repository contains the code for retinal vessel segmentation research. The code includes data preprocessing, multi-source pre-training on STARE/DRIVE/CHASE_DB1/HRF, and fine-tuning/ablation studies on the FIVES dataset.
Project Structure
processing.py - Data preprocessing codeunet.py - U-Net model and multi-source pre-training codeablation.py - FIVES fine-tuning and ablation study coderequirement.txt - Python environment dependencies
Quick Start
Set up the environment:conda create -n retinal-seg python=3.8conda activate retinal-segpip install -r requirement.txt
Prepare the datasets:Download STARE, DRIVE, CHASE_DB1, HRF, and FIVES from their official websites.Place the raw images and annotations in the correct folders (paths can be modified in the code).
Run the code:python processing.pypython unet.pypython ablation.py
Key Results
The best performance on FIVES test set is 0.8971 ± 0.1022 Dice coefficient (using green channel extraction + CLAHE, no TTA).
Citation
If you use this code, please cite our paper:@article{yourname2026retinal,title={Enhancing Generalization in Retinal Vessel Segmentation: A Transfer Learning Framework with Robust Domain Calibration},author={Your Name},journal={Your Journal},year={2026}}
License
This project is licensed under the MIT License.