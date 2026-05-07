# traffic_signs_image_classification

PyTorch Traffic Sign Classifier: Custom CNN vs. ResNet18
Overview

This project explores computer vision and deep learning by building neural networks to classify the German Traffic Sign Recognition Benchmark (GTSRB) dataset into 43 distinct classes. The project demonstrates a full deep learning pipeline in PyTorch, from dataset loading and image transformations to model training, evaluation, and inference.
Project Highlights

This repository contains two different approaches to the same image classification problem:

1. Custom Convolutional Neural Network (Baseline)

    A CNN built entirely from scratch (TrafficSignNet) using PyTorch.

    Utilizes a 3-layer convolutional architecture with Max Pooling, Dropout (to prevent overfitting), and Fully Connected layers.

    Processes raw 32x32 images.

    Performance: Achieved an official test set accuracy of 94.06%.

2. Transfer Learning with ResNet18

    Adapts a pre-trained Microsoft ResNet18 model (originally trained on ImageNet) to classify traffic signs.

    Demonstrates professional computer vision techniques including:

        Freezing pre-trained convolutional backbones (feature extractors).

        Replacing the final fully connected layer for a custom 43-class output.

        Image scaling (224x224) and mathematical Color Normalization matching ImageNet standards.

        Targeted optimization using a conservative learning rate (0.0001).

Tech Stack

    Language: Python

    Framework: PyTorch & Torchvision

    Data Processing: Pandas, PIL (Python Imaging Library)

    Visualization: Matplotlib

Repository Structure

    train.ipynb - Training loop for the baseline CNN.

    evaluate.ipynb - Testing and inference for the baseline model.

    transfer_learning.ipynb - Training pipeline for the pre-trained ResNet18 model.

    evaluate_resnet.ipynb - Testing and inference for the Transfer Learning model.
