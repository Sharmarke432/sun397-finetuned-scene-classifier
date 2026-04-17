# SUN397 Finetuned Scene Classifier

This project fine-tunes three pretrained convolutional neural networks (**EfficientNet-B0**, **MobileNetV3**, and **ResNet18**) on 10 selected SUN397 scene categories using transfer learning, then evaluates and compares their performance on held-out test data.

## Project Overview

The goal is to build a robust scene classifier on a subset of the SUN397 dataset by:

- Selecting 10 scene categories (e.g., beach, bedroom, kitchen, office). 
- Loading SUN397 via `torchvision.datasets.SUN397` and/or Hugging Face `datasets`. 
- Using three pretrained backbones: **EfficientNet-B0**, **MobileNetV3**, and **ResNet18** from `timm` / `torchvision`.
- Fine-tuning these models on the 10 classes with PyTorch. 
- Providing scripts for training, evaluation, and inference on custom images.

This setup demonstrates practical transfer learning with lightweight, widely used backbones suitable for real-world scene classification.

## Dataset

We use the **SUN397** scene dataset, a large-scale benchmark for scene recognition. 

Key facts:

- 397 scene categories.  
- At least 100 images per category.  
- Over 108,000 images total. 

In this project we restrict to 10 scene categories, for example:

```python
CLASSES = [
    "beach",
    "bedroom",
    "kitchen",
    "living room",
    "office",
    "mountain",
    "highway",
    "street",
    "church indoor",
    "forest broadleaf",
]
```

### Dataset access

- Load SUN397 with `torchvision.datasets.SUN397` for a PyTorch-native pipeline.
- Optionally load a SUN397-style dataset from Hugging Face via `datasets.load_dataset` for experimentation or alternative splits.
- Use `pandas`/`numpy` for metadata inspection and filtering.  

You then filter down to the 10 selected classes and create train/validation/test splits.

## Dependencies

The project uses the following main libraries (reflecting the imports used in the notebooks / scripts):

- **Core ML / DL**
  - `torch`, `torchvision`, `torchvision.models`, `torch.utils.data`
  - `timm` (EfficientNet-B0, MobileNetV3, other pretrained models) 
  - `torch.nn`  
- **Datasets and data utilities**
  - `torchvision.datasets.SUN397` 
  - `datasets` from Hugging Face (`load_dataset`, `DatasetDict`, `load_from_disk`)
  - (optionally) `tensorflow_datasets` and `tensorflow` for alternative data loading / comparison  
- **General utilities**
  - `pandas`, `numpy`  
  - `PIL` (Pillow) for image handling  
  - `matplotlib.pyplot` for visualization  
  - `os` for filesystem handling  
- **Colab / Hub integration**
  - `google.colab.files` for image upload in notebooks  
  - `huggingface_hub.notebook_login` for authenticating and pushing models/datasets to the Hub 

Install (minimal) via:

```bash
pip install torch torchvision timm datasets pandas numpy matplotlib
```

## Models

We fine-tune three pretrained backbones:

- **EfficientNet-B0**
  - Loaded from `timm` with ImageNet-pretrained weights.   
  - Parameter-efficient CNN that scales depth/width/resolution with a compound coefficient. 

- **MobileNetV3**
  - Loaded from `timm` with ImageNet-pretrained weights. 
  - Lightweight model designed for mobile and edge devices.  

- **ResNet18**
  - Loaded from `torchvision.models` with ImageNet-pretrained weights.
  - Classic residual architecture widely used for transfer learning.  

Transfer learning steps for each backbone:

- Load the ImageNet-pretrained model.
- Replace the final classification layer with a new layer outputting 10 logits.   
- Optionally freeze early layers and fine-tune only the classifier head or last few blocks.  

All three models are trained separately on the same 10-class SUN397 subset.

## End-to-End Pipeline

High-level pipeline:

### 1. Data preparation

- Download / load SUN397 using `torchvision.datasets.SUN397` (and optionally Hugging Face `datasets`).   
- Define the list of 10 target scene classes.  
- Filter the dataset to keep only examples whose labels match these 10 classes.  
- Split the filtered dataset into training, validation, and test sets (e.g., stratified split per class).  
- Define image transforms (resize, data augmentation, normalization) using `torchvision.transforms`.  
- Wrap splits into `DataLoader` objects for batched training and evaluation.

### 2. Model setup

- For each backbone (EfficientNet-B0, MobileNetV3, ResNet18):
  - Load the pretrained model (ImageNet weights). 
  - Replace the final classification layer with a new layer for 10 classes.  
  - Move the model to the selected device (CPU / GPU).  
  - Configure loss function (cross-entropy) and optimizer (e.g., Adam/AdamW/SGD).

### 3. Training loop

- For each epoch:
  - Put the model in training mode.  
  - Iterate over training batches:
    - Load image batch and labels, move to device.  
    - Forward pass through the model to compute logits.  
    - Compute cross-entropy loss.  
    - Backpropagate gradients and update parameters.  
  - Optionally evaluate on the validation set at the end of each epoch.  
  - Track metrics (loss, accuracy) and save the best checkpoints for each model.  

### 4. Evaluation

- Load the best checkpoint for each backbone.  
- Put the model in evaluation mode.  
- Iterate over the test set:
  - Compute predictions and probabilities.  
  - Accumulate metrics (overall accuracy, per-class accuracy).  
- Optionally:
  - Compute a confusion matrix for the 10 classes.  
  - Build a simple ensemble by averaging logits from all three models and re-running evaluation.

### 5. Inference on custom images

- Load a single image (e.g., with `PIL.Image.open`).  
- Apply the same transforms used during training (resize, normalization).  
- Add batch dimension and move to device.  
- Run a forward pass through a fine-tuned model (or the ensemble).  
- Apply softmax to obtain class probabilities and pick the top-1 or top-k predictions.  
- Print or visualize the predicted scene label.

## Repository Structure

A suggested layout:

```text
.
├── data/
├── src/
│   ├── dataset.py        # Dataset utilities and filtering for 10 classes
│   ├── models.py         # EfficientNet-B0, MobileNetV3, ResNet18 wrappers
│   ├── train.py          # Training loops for each backbone
│   ├── evaluate.py       # Evaluation and metrics
│   └── inference.py      # Inference utilities / CLI
├── notebooks/
│   └── exploration.ipynb
├── checkpoints/
├── requirements.txt
└── README.md
```

## References

- SUN397 scene recognition benchmark.
- Transfer learning tutorial with ResNet18 (PyTorch).  
- `timm` – PyTorch Image Models (EfficientNet-B0, MobileNetV3, etc.). 
- Example transfer learning implementations with ResNet18. 
