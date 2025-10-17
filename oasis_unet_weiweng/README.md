# OASIS Brain MRI Segmentation using Improved U-Net  
**Author:** Wei Weng (Student ID: 47981739)  
**Course:** COMP3710 – Pattern Analysis (The University of Queensland)  
**Year:** 2025  

---

## 🧠 Overview
This project implements a **2D Improved U-Net** model to perform brain MRI segmentation using the **OASIS** dataset.  
The objective is to achieve a **Dice Similarity Coefficient (DSC) ≥ 0.9** across all segmentation labels on the test set.  

The model is built with **PyTorch** and designed following professional open-source collaboration workflows.  
It contributes to the **PatternAnalysis-2025** library through a pull request under the `topic-recognition` branch.

---

## 🎯 Problem Definition
Brain MRI segmentation is an essential step in medical image analysis, allowing for the identification of brain structures and abnormalities.  
The task involves assigning a class label to each pixel of the MRI slice to delineate different anatomical regions.

**Dataset:**  
- **Name:** OASIS Brain MRI (2D version)  
- **Source:** /home/groups/comp3710/OASIS (Rangpur HPC)  
- **Labels:** Multi-region brain structures  
- **Format:** NIfTI (`.nii` / `.nii.gz`)

---

## 🧩 Model Description
The model is based on **U-Net (Ronneberger et al., 2015)**, with several enhancements:

| Component | Description |
|------------|--------------|
| Encoder | Convolutional blocks with BatchNorm + ReLU activation |
| Decoder | Transposed convolution for upsampling |
| Improvements | Added dropout layers and deeper skip connections |
| Loss Function | Dice Loss + Cross Entropy Loss |
| Optimizer | Adam (lr = 1e-4) |
| Metrics | Dice Similarity Coefficient (DSC), SSIM |

---

## 🧰 File Structure
