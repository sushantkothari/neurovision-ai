<div align="center">

<h1>NeuroVision AI</h1>
<h3>Brain Tumor Classification and Segmentation on BraTS</h3>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Medical%20Imaging-0A9396?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Architecture-EfficientNet%20%2B%20Residual%20Attention%20U--Net-7B2FBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-2D6A4F?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-BraTS%202020-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tasks-Classification%20%7C%20Segmentation-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Modality-MRI-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Project-Research%20Grade-red?style=flat-square"/>
</p>

<br/>

> **A research-grade deep learning pipeline for automated brain tumor classification and tumor-region segmentation from BraTS MRI data, combining CNN-based classification with a Residual Attention U-Net segmentation backbone for end-to-end medical image analysis.**

</div>

---

# Table of Contents

- [Project Overview](#project-overview)
- [Why This Problem Matters](#why-this-problem-matters)
- [Technical Highlights](#technical-highlights)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Dataset](#dataset)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Training Configuration](#training-configuration)
- [Evaluation Framework](#evaluation-framework)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Inference and Saved Artifacts](#inference-and-saved-artifacts)
- [Technology Stack](#technology-stack)
- [Engineering Principles](#engineering-principles)
- [Potential Extensions](#potential-extensions)
- [Limitations](#limitations)
- [License](#license)
- [Author](#author)

---

# Project Overview

This repository implements a complete deep learning workflow for **brain tumor analysis** using MRI scans from the **BraTS benchmark dataset**. The system is structured around two complementary tasks:

1. **Binary Classification**
   - MRI slice classification into `Normal` vs `Tumor`

2. **Tumor Segmentation**
   - Pixel-level segmentation of tumor subregions from multimodal MRI scans

The project combines advanced deep learning architectures, medical image preprocessing, augmentation pipelines, overlap-aware loss functions, evaluation metrics, and qualitative visualization into a single reproducible notebook pipeline.

The repository is designed as a **research-grade yet recruiter-friendly medical imaging project**, suitable for:
- GitHub portfolios
- Resume projects
- Research demonstrations
- Deep learning showcases
- Medical AI experimentation

---

# Why This Problem Matters

Brain tumor analysis is one of the most impactful applications of artificial intelligence in healthcare.

## Early Detection

MRI interpretation is time-consuming and highly specialized. Automated tumor classification systems can assist radiologists by identifying suspicious scans more rapidly and consistently.

## Tumor Localization

Classification alone is insufficient in real-world clinical workflows. Accurate segmentation is critical because it identifies:
- tumor boundaries
- abnormal tissue regions
- tumor extent
- spatial progression

This information directly supports:
- surgical planning
- radiotherapy targeting
- longitudinal monitoring
- treatment evaluation

## Research Significance

The BraTS benchmark is one of the most widely used datasets in brain tumor segmentation research, making it an excellent platform for benchmarking modern deep learning architectures.

---

# Technical Highlights

## Architecture

- CNN-based tumor classification pipeline
- Residual Attention U-Net segmentation architecture
- Encoder-decoder feature fusion
- Attention-gated skip connections
- Multi-stage medical imaging workflow

## Medical Imaging Pipeline

- BraTS MRI preprocessing workflow
- `.h5` slice-based MRI handling
- Metadata-driven dataset organization
- MRI mask processing and visualization
- Structured train / validation / test handling

## Training Robustness

- Dice-based segmentation optimization
- BCE-enhanced overlap supervision
- Validation-based metric monitoring
- Qualitative prediction visualization
- Organized experiment workflow

## Engineering Quality

- Modular notebook organization
- Clean repository structure
- Reproducible execution pipeline
- GitHub-ready presentation formatting
- Saved inference outputs and model artifacts

---

# Architecture Deep Dive

## Conceptual Workflow

```text
Raw BraTS MRI Data
        |
        v
+-----------------------------------------------+
|              PREPROCESSING STAGE              |
|  - Load MRI metadata                          |
|  - Read `.h5` MRI slice volumes               |
|  - Normalize and organize tensors             |
|  - Generate segmentation masks                |
+-------------------+---------------------------+
                    |
                    v
+-----------------------------------------------+
|        STAGE 1 -- CLASSIFICATION MODEL        |
|  CNN Feature Extractor                        |
|  EfficientNet-style Backbone                  |
|  Dense Classification Head                    |
|  Output: Normal vs Tumor                      |
+-------------------+---------------------------+
                    |
                    v
+-----------------------------------------------+
|        STAGE 2 -- SEGMENTATION MODEL          |
|  Residual Attention U-Net                     |
|  Encoder-Decoder Architecture                 |
|  Attention-Gated Skip Connections             |
|  Multi-Region Tumor Segmentation              |
+-------------------+---------------------------+
                    |
                    v
+-----------------------------------------------+
|                 EVALUATION                    |
|  - Accuracy / Precision / Recall / F1         |
|  - Dice Score / IoU / Pixel Accuracy          |
|  - Qualitative MRI Visualization              |
+-----------------------------------------------+
```

## Why Classification and Segmentation Together?

Classification answers:

> “Does this MRI slice contain a tumor?”

Segmentation answers:

> “Where exactly is the tumor located?”

Combining both creates a much more complete medical imaging pipeline compared to standalone classifiers.

## Why Residual Attention U-Net?

Residual Attention U-Net improves traditional U-Net segmentation by:

- enhancing gradient propagation
- improving feature reuse
- emphasizing tumor-relevant regions
- suppressing irrelevant anatomical background

This is especially important for MRI scans where tumors may:
- occupy small regions
- have irregular structure
- blend with surrounding tissue

---

# Dataset

## BraTS 2020 Dataset

The project is built around the **BraTS 2020 benchmark dataset**, a widely used benchmark for brain tumor segmentation research.

## Expected Files

The notebook expects:

- `meta_data.csv`
- `name_mapping.csv`
- `survival_info.csv`
- all `volume_*_slice_*.h5` files

## Expected Dataset Structure

```text
data/raw/brats2020/BraTS2020_training_data/content/data/
```

## Dataset Notes

- Metadata CSV files are included in the repository
- Raw MRI `.h5` files are excluded from GitHub due to size limitations
- The notebook follows a BraTS-style MRI organization workflow

---

# End-to-End Pipeline

```text
Step 1   Load BraTS MRI metadata and `.h5` files

Step 2   Preprocess MRI slices
         |- Normalize image tensors
         |- Prepare segmentation masks
         +- Organize labels

Step 3   Create train / validation / test splits

Step 4   Train classification model
         |- CNN backbone
         +- Binary tumor classification

Step 5   Train segmentation model
         |- Residual Attention U-Net
         +- Tumor-region segmentation

Step 6   Evaluate classification
         |- Accuracy
         |- Precision
         |- Recall
         +- F1 Score

Step 7   Evaluate segmentation
         |- Validation Loss
         |- Dice Score
         |- IoU
         +- Pixel Accuracy

Step 8   Save outputs
         |- Model weights
         |- Metrics CSV
         +- Qualitative predictions
```

---

# Machine Learning Methodology

## Classification Pipeline

The classification stage uses a CNN-based feature extraction backbone for identifying tumor-bearing MRI slices.

CNN architectures are highly effective for MRI classification because they capture:
- local texture patterns
- structural asymmetry
- tumor-specific visual signatures

## Segmentation Pipeline

The segmentation stage predicts tumor masks at the pixel level using a Residual Attention U-Net.

The architecture combines:
- encoder-decoder reconstruction
- residual learning
- attention-based feature refinement
- high-resolution spatial recovery

## Loss Design

Medical segmentation suffers heavily from class imbalance because tumor regions often occupy a very small portion of the image.

To address this:
- Dice-based objectives optimize overlap quality
- BCE-style losses improve pixel-level supervision

This hybrid strategy improves segmentation stability and mask quality.

---

# Training Configuration

| Parameter | Value |
|---|---|
| Classification Task | Binary Tumor Detection |
| Segmentation Task | Multi-Region Tumor Segmentation |
| Classification Backbone | EfficientNet-style CNN |
| Segmentation Backbone | Residual Attention U-Net |
| Input Modality | MRI |
| Input Format | `.h5` MRI slices |
| Loss Functions | Dice + BCE-style losses |
| Metrics | Accuracy, Precision, Recall, F1, Dice, IoU |
| Framework | PyTorch |
| Notebook Environment | JupyterLab / Notebook |

---

# Evaluation Framework

## Classification Results

| Metric | Score |
|---|---|
| Validation Accuracy | `0.9730` |
| Precision | `0.9771` |
| Recall | `0.9599` |
| F1 Score | `0.9684` |

## Segmentation Results

| Metric | Score |
|---|---|
| Validation Loss | `0.2088` |
| Dice Score | `0.7959` |
| IoU | `0.7300` |
| Pixel Accuracy | `0.9982` |

These results demonstrate strong performance across both classification and segmentation tasks.

---

# Repository Structure

```text
neurovision-ai-brain-tumor-detection/
|
+-- notebooks/
|   +-- Hybrid_DeepLearning_Framework_for_Brain_Tumor_Classification_and_Segmentation.ipynb
|
+-- data/
|   +-- metadata/
|   |   +-- meta_data.csv
|   |   +-- name_mapping.csv
|   |   +-- survival_info.csv
|   |
|   +-- raw/
|       +-- brats2020/
|           +-- BraTS2020_training_data/
|               +-- content/
|                   +-- data/
|                       +-- volume_*_slice_*.h5
|
+-- models/
|   +-- classifier_weights.pth
|   +-- segmentation_weights.pth
|
+-- results/
|   +-- metrics.csv
|   +-- qualitative_predictions/
|
+-- assets/
|   +-- project_figures/
|
+-- environment.yml
+-- requirements.txt
+-- README.md
```

---

# Quickstart

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/neurovision-ai-brain-tumor-detection.git
cd neurovision-ai-brain-tumor-detection
```

## 2. Install Dependencies

You can set up the environment using Conda (Recommended):

```bash
conda env create -f environment.yml
conda activate neurovision-ai
```

Or using pip:

```bash
pip install -r requirements.txt
```

## 3. Prepare Dataset

Place the BraTS MRI files under:

```text
data/raw/brats2020/BraTS2020_training_data/content/data/
```

## 4. Launch JupyterLab

```bash
jupyter lab
```

## 5. Run the Notebook

Open:

```text
notebooks/Hybrid_DeepLearning_Framework_for_Brain_Tumor_Classification_and_Segmentation.ipynb
```

Run all cells sequentially from top to bottom.

---

# Inference and Saved Artifacts

## Saved Outputs

The repository includes or generates:

- trained classifier weights
- trained segmentation weights
- saved metrics
- qualitative MRI predictions
- segmentation visualizations

## Inference Workflow

1. Load trained weights
2. Preprocess MRI slices
3. Run classification inference
4. Run segmentation inference
5. Visualize predicted masks

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core runtime |
| PyTorch | Deep learning framework |
| Torchvision | Vision utilities |
| Timm | Pretrained CNN backbones |
| Albumentations | Image augmentation |
| OpenCV | Image preprocessing |
| Pillow | Image I/O |
| Scikit-image | Medical image utilities |
| H5py | MRI `.h5` file handling |
| NumPy | Numerical operations |
| Pandas | Metadata processing |
| Matplotlib | Visualization |
| Scikit-learn | Metrics and evaluation |
| JupyterLab | Interactive notebook environment |

---

# Engineering Principles

## Task Specialization

The project separates:
- tumor detection
- tumor localization

into dedicated pipelines for cleaner optimization and evaluation.

## Architectural Fit

CNNs are optimized for image-level classification, while U-Net architectures are specifically designed for dense biomedical segmentation.

## Medical Imaging Awareness

Attention modules and overlap-aware losses are used because tumor regions are often:
- small
- irregular
- highly imbalanced relative to background pixels

## Reproducibility

The repository structure separates:
- code
- data
- models
- outputs
- assets

to maintain organization and reproducibility.

---

# Potential Extensions

## Architecture Improvements

- Vision Transformers for classification
- Attention-enhanced decoder blocks
- Ensemble segmentation systems
- Multi-scale feature fusion

## Research Extensions

- Cross-validation experiments
- Multi-dataset generalization
- Advanced augmentation policies
- Hyperparameter optimization

## Deployment Targets

- TorchScript export
- ONNX conversion
- FastAPI inference server
- Lightweight inference pipelines

---

# Limitations

- This repository is intended for research and educational purposes
- It is not a clinically validated diagnostic system
- Performance on BraTS does not guarantee real-world hospital deployment performance
- Clinical deployment requires regulatory approval and expert supervision

---

# License

This project is licensed under the MIT License.

---

# Author

**Sushant Kothari**

GitHub: https://github.com/<your-username>

---

<div align="center">

If this project was useful or informative, consider starring the repository on GitHub.

</div>
