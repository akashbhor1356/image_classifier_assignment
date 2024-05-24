# Image Classification with Handcrafted Features and Flask

This project demonstrates image classification using handcrafted features and a shallow learning model (SVM). The project includes feature extraction, model training, evaluation, and deployment through a Flask web application.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Flask App](#running-the-flask-app)
- [Evaluation](#evaluation)
- [Enhancements](#enhancements)
- [License](#license)

## Dataset

The dataset used for this project contains images classified into six categories:
- Buildings
- Forest
- Glacier
- Mountains
- Sea
- Streets

The dataset can be downloaded from [this link](https://drive.google.com/file/d/1lWgKokYUrD5PPMO3tCy-yMN2ytaSnjTV/view?usp=sharing).

## Features

The following handcrafted features were extracted from the images:
- Histogram Features
- Edge Features (using Canny edge detection)

## Model

A Support Vector Machine (SVM) with a linear kernel was used for classification. Dimensionality reduction was performed using PCA.

## Requirements

To run this project, you need the following libraries:

- Python 3.8+
- Flask
- OpenCV
- NumPy
- scikit-learn
- joblib

Install the required packages using pip:

```bash
pip install flask opencv-python-headless numpy scikit-learn joblib
# image_classifier_assignment
