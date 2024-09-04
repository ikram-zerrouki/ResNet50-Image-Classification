# Project Name: **ResNet50 Image Classification**

## Description

This project showcases the fine-tuning and training of the ResNet50 model for binary image classification using TensorFlow and Keras. The ResNet50 architecture is known for its deep layers and residual learning, making it suitable for complex image recognition tasks.

The dataset is split into three subsets:
- **70% for training**
- **10% for validation**
- **20% for testing**

These ratios can be customized as needed.

## Features

- **Data Preprocessing**: Includes image resizing, augmentation, and normalization using the ResNet50 preprocessing function.
- **Model Architecture**: Built on top of the ResNet50 base model with additional layers for binary classification.
- **Training Process**: Integrated with early stopping and model checkpointing to ensure optimal performance.
- **Evaluation**: Provides accuracy, loss plots, and confusion matrix visualization.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Required Libraries: `numpy`, `matplotlib`, `seaborn`

## Ensure your dataset is organized as follows :
data/
├── Training_Dataset/
│   ├── Class_0/
│   └── Class_1/
├── Validation_Dataset/
│   ├── Class_0/
│   └── Class_1/
└── Test_Dataset/
    ├── Class_0/
    └── Class_1/
