# 🔬 Image Classification Using VGG-19 Model with Streamlit

A complete academic project that demonstrates **deep learning-based image classification**
using the **VGG-19** convolutional neural network pretrained on ImageNet (1 000 classes),
packaged as an interactive **Streamlit** web application.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Installation & Setup (Windows)](#installation--setup-windows)
6. [Running the Application](#running-the-application)
7. [Application Workflow](#application-workflow)
8. [Model Architecture – VGG-19](#model-architecture--vgg-19)
9. [Preprocessing Pipeline](#preprocessing-pipeline)
10. [Limitations](#limitations)
11. [Future Enhancements](#future-enhancements)
12. [Viva Questions & Answers](#viva-questions--answers)
13. [References](#references)

---

## Project Overview

This project showcases how a state-of-the-art pretrained CNN (VGG-19) can be integrated into
a beginner-friendly web interface. Users upload any image and the app returns the top-5
predicted object categories with confidence scores, a sortable table, and a bar chart —
all without requiring custom model training.

> **Note:** This application uses the **officially pretrained ImageNet weights** provided
> by `torchvision`. No custom training was performed.

---

## Features

| Feature | Details |
|---------|---------|
| Image upload | JPG, JPEG, PNG support |
| Real-time inference | VGG-19 forward pass on CPU |
| Top-1 prediction | Displayed as a highlighted badge |
| Top-5 predictions | Interactive table + bar chart |
| Image metadata | Dimensions, colour mode, file size |
| Model caching | `st.cache_resource` — loads once |
| Error handling | Invalid files & runtime errors caught |
| Responsive UI | Two-column layout, custom CSS |
| Sidebar | Project info, model details, workflow |

---

## Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| Streamlit | 1.35.0 | Web UI framework |
| PyTorch | 2.3.1 | Deep learning inference |
| Torchvision | 0.18.1 | Pretrained VGG-19 & transforms |
| Pillow | 10.3.0 | Image loading & validation |
| Pandas | 2.2.2 | Results table |

---

## Project Structure

```
VGG19_Image_Classification/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies (pinned versions)
├── README.md         ← This file
└── .gitignore        ← Git ignore rules
```

---

## Installation & Setup (Windows)

### Prerequisites

- Python 3.10 or newer  
- `pip` (bundled with Python)  
- Internet connection (first run downloads the VGG-19 weights, ~548 MB)

### Step-by-step (Windows CMD)

```cmd
:: 1. Navigate to the project folder
cd "d:\Web Projects\LangChain Projects\VGG19_Image_Classification"

:: 2. Create a virtual environment
python -m venv venv

:: 3. Activate the virtual environment
venv\Scripts\activate

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install all dependencies
pip install -r requirements.txt
```

### Step-by-step (Windows PowerShell)

```powershell
# 1. Navigate to the project folder
Set-Location "d:\Web Projects\LangChain Projects\VGG19_Image_Classification"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt
```

> **Tip for PowerShell users:** if you see an execution-policy error, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.

---

## Running the Application

```cmd
:: Make sure the virtual environment is active, then:
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).
Open it in any modern browser.

**First-run note:** PyTorch will automatically download the VGG-19 pretrained weights
(~548 MB) on the very first launch and cache them locally. Subsequent starts are fast.

---

## Application Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER UPLOADS IMAGE                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Pillow validates & opens the image (RGB conversion)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  VGG19_Weights.DEFAULT.transforms() preprocessing:          │
│  • Resize shortest edge → 256 px                            │
│  • Centre-crop → 224 × 224 px                               │
│  • ToTensor  (pixel values 0–1)                             │
│  • Normalise with ImageNet mean & std                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  VGG-19 forward pass  (16 Conv + 3 FC layers)               │
│  Output: logit tensor of shape (1, 1000)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Softmax → probability distribution over 1 000 classes      │
│  torch.topk(probs, 5) → top-5 (label, confidence) pairs     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit renders:                                          │
│  • Uploaded image + metadata                                 │
│  • Top-1 badge with confidence %                             │
│  • Top-5 table (Pandas DataFrame)                            │
│  • Bar chart (st.bar_chart)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture – VGG-19

VGG-19 was proposed by **Simonyan & Zisserman (2014)** from the Visual Geometry Group,
University of Oxford. It won 2nd place in the ILSVRC 2014 classification challenge.

```
INPUT  224×224×3
  │
  ├── Block 1:  Conv3-64  × 2  →  MaxPool  →  112×112×64
  ├── Block 2:  Conv3-128 × 2  →  MaxPool  →   56×56×128
  ├── Block 3:  Conv3-256 × 4  →  MaxPool  →   28×28×256
  ├── Block 4:  Conv3-512 × 4  →  MaxPool  →   14×14×512
  ├── Block 5:  Conv3-512 × 4  →  MaxPool  →    7×7×512
  │
  ├── Flatten  →  25 088
  ├── FC-4096  +  ReLU  +  Dropout
  ├── FC-4096  +  ReLU  +  Dropout
  └── FC-1000  →  Softmax  →  OUTPUT (1 000 class probabilities)

Total trainable parameters: ~143.7 million
```

---

## Preprocessing Pipeline

Torchvision's `VGG19_Weights.DEFAULT.transforms()` applies the **exact same pipeline**
used during the model's original ImageNet training:

| Step | Operation | Value |
|------|-----------|-------|
| 1 | Resize | Shorter edge → 256 px |
| 2 | CentreCrop | 224 × 224 px |
| 3 | ToTensor | Pixel range 0–1 |
| 4 | Normalise mean | [0.485, 0.456, 0.406] |
| 5 | Normalise std | [0.229, 0.224, 0.225] |

Using the wrong normalisation values is a common bug; `VGG19_Weights.DEFAULT.transforms()`
guarantees correctness automatically.

---

## Limitations

- **Fixed vocabulary:** Only recognises 1 000 ImageNet categories; unknown objects will
  be mis-classified as the closest known category.  
- **Resolution sensitivity:** Very small or very large input images are forcibly resized,
  potentially losing detail.  
- **CPU speed:** VGG-19 is large (~550 MB); each inference on CPU takes 1–5 seconds.  
- **No explainability:** The app does not show *why* a prediction was made (no Grad-CAM).  
- **No domain adaptation:** Not suitable for medical imaging, satellite imagery, or other
  specialised domains without fine-tuning.

---

## Future Enhancements

- **Grad-CAM heatmaps** – visualise which image regions drove the prediction  
- **Model comparison** – benchmark VGG-19 vs. ResNet-50 vs. EfficientNet-B0  
- **GPU acceleration** – detect CUDA and move tensors to GPU automatically  
- **Fine-tuning panel** – allow uploading a small custom dataset and fine-tuning the FC layers  
- **Batch upload** – classify multiple images at once  
- **Confidence threshold** – reject predictions below a user-defined confidence  
- **Streamlit Cloud deployment** – one-click public sharing  

---

## Viva Questions & Answers

### Q1. What is VGG-19 and why was it chosen for this project?
**A:** VGG-19 is a deep convolutional neural network with 19 weight layers (16 convolutional
+ 3 fully connected) developed by the Visual Geometry Group at Oxford University in 2014.
It was chosen because it uses a simple, uniform architecture (only 3×3 convolutions), has
excellent pretrained ImageNet weights available in Torchvision, and is widely studied in
academic literature, making it ideal for demonstrating CNN-based image classification.

### Q2. What is ImageNet and how many classes does VGG-19 recognise?
**A:** ImageNet is a large-scale visual database with over 14 million labelled images.
The subset used for the ILSVRC challenge (ImageNet Large Scale Visual Recognition Challenge)
contains 1.2 million training images across **1 000 categories**. VGG-19 trained on this
dataset can recognise 1 000 different object categories.

### Q3. Why do we apply normalisation before passing the image to the model?
**A:** Neural networks converge faster and perform better when input features have a similar
scale. The normalisation values (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
were computed from the entire ImageNet training set. Applying the **same** normalisation at
inference time ensures the input distribution matches what the model saw during training.

### Q4. What does `VGG19_Weights.DEFAULT` mean in Torchvision?
**A:** `VGG19_Weights.DEFAULT` is an enum that points to the best available pretrained
weights for VGG-19 in Torchvision — currently `VGG19_Weights.IMAGENET1K_V1`. It also
exposes the associated `transforms()` method, which returns the exact preprocessing pipeline
used during training. Using `.DEFAULT` is the recommended modern API (deprecated `pretrained=True`
parameter is avoided).

### Q5. What is Softmax and why is it used in the output layer?
**A:** Softmax converts a vector of raw logits (unbounded real numbers) into a **probability
distribution** — all values are in [0, 1] and sum to 1. This makes the model's output
directly interpretable as class confidence scores. The class with the highest probability is
the predicted label.

### Q6. What is `st.cache_resource` and why is it used?
**A:** `st.cache_resource` is a Streamlit decorator that caches objects shared across all
users and sessions — perfect for ML models. Without it, the VGG-19 model (~548 MB) would be
reloaded from disk on every user interaction, making the app extremely slow. With caching,
the model is loaded once and reused for all subsequent inference calls.

### Q7. How does the app handle corrupt or non-image files?
**A:** The app calls `Image.verify()` from Pillow, which reads the file header and raises an
`UnidentifiedImageError` or a generic exception if the file is truncated or not a valid image.
Streamlit then catches these exceptions and displays a user-friendly error message instead of
crashing.

### Q8. What is the difference between a Convolutional layer and a Fully Connected layer?
**A:** A **convolutional layer** applies learnable filters (kernels) that slide over the
spatial dimensions of the input, preserving spatial structure and sharing weights across
positions — making it translation-invariant and parameter-efficient. A **fully connected
(FC) layer** connects every input neuron to every output neuron, treating the input as a
flat vector. In VGG-19, convolutional layers extract spatial features, while FC layers
perform final classification.

### Q9. Why does VGG-19 use 3×3 convolutions exclusively?
**A:** Simonyan & Zisserman showed that stacking two 3×3 conv layers gives the same
**receptive field** as one 5×5 layer, but with **fewer parameters** (2 × 3² = 18 vs. 25
weights per channel) and an extra non-linearity (ReLU), allowing the network to learn
more complex features. Using only small 3×3 kernels also makes the architecture uniform
and easy to analyse.

### Q10. Can this application classify images outside the 1 000 ImageNet categories?
**A:** No — VGG-19 can only output one of the 1 000 categories it was trained on. If an
image does not belong to any of those categories, the model will still force-assign the
closest class (possibly with a low confidence score). To handle new categories, the model
would need to be **fine-tuned** on a custom dataset with the new classes.

---

## References

1. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale
   Image Recognition.* arXiv:1409.1556. <https://arxiv.org/abs/1409.1556>  
2. PyTorch Documentation – `torchvision.models.vgg19`:
   <https://pytorch.org/vision/stable/models/vgg.html>  
3. Streamlit Documentation: <https://docs.streamlit.io>  
4. Deng, J. et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.*
   IEEE CVPR 2009.  
5. ImageNet Simple Labels (class names used in this project):
   <https://github.com/anishathalye/imagenet-simple-labels>  

---

*This project was built for academic demonstration purposes using publicly available
pretrained weights. No custom model training was performed.*
