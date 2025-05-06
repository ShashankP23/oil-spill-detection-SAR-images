# Oil Spill Detection from SAR Satellite Images

This project implements a convolutional neural network (CNN) in PyTorch to detect the presence of oil spills from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. The model performs binary classification to distinguish between oil slick and non-oil slick images.

## Dataset

The dataset is provided by CSIRO and consists of SAR images labeled as either oil (_cls_1) or non-oil (_cls_0). Images are grayscale and organized in an unbalanced format under the `data/` directory.

Dataset link: [CSIRO SAR Oil Images](https://data.csiro.au/collection/csiro:57430)

## Model

- Architecture: Custom CNN with 3 convolutional layers, ReLU activations, max pooling, and fully connected layers
- Loss Function: `BCEWithLogitsLoss` for stable binary classification
- Optimizer: Adam (learning rate = 0.001)
- Input size: 400x400 single band (grayscale) images

## Training

- Binary classification between oil slick and non-oil slick images
- Trained for 20 epochs with performance tracked using running loss
- Used GPU acceleration via Google Colab

## Evaluation

Evaluation is run from a small, separate sample (for simplicity); a more robust setup would involve train/test split.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- PIL (Pillow)
- scikit-learn

Install with:

```bash
pip install torch torchvision scikit-learn Pillow matplotlib
```