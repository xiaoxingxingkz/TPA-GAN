# Task-Induced Pyramid and Attention GAN for Multimodal Brain Image Imputation and Classification in Alzheimer’s Disease
Paper: https://ieeexplore.ieee.org/document/9490307

## Summary

In practice, multimodal images may be incomplete since PET is often missing due to high financial costs or availability. Most of the existing methods simply excluded subjects with missing data, which unfortunately reduced the sample size. To address these problems, we propose a task-induced pyramid and attention generative adversarial network (TPA-GAN) for imputation of multimodal brain images. With the complete multimodal images, we build a pathwise transfer dense convolution network (PT-DCN) to gradually learn and combine the multimodal features for final disease classification.

## Overview
<p align="center">
  <img src="https://github.com/xiaoxingxingkz/TPA-GAN/blob/main/Figure_in_paper/Fig1.png" width="700">
</p>

## Installation

This script need no installation, but has the following requirements:
* PyTorch 1.11.0 or above
* Python 3.5.7 or above

## Usage

Import libraries

```python
import os
import cv2
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
import math
import time
```
## Methods
### Stage 1: TPA-GAN for multimodal brain image imputation
<p align="center">
  <img src="https://github.com/xiaoxingxingkz/TPA-GAN/blob/main/Figure_in_paper/Fig2.png" width="700">
</p>

Run the following code，we test the models after each epoch of training process. No independent test program is required, we also provide。
```python
train_TPA_GAN.py
```
### Tricks
* Pre-train the Task-induced discriminator first. In the training of GAN, the weights of Task-induced discriminator are fixed, while its loss is used to update the parameters of generator, which can help the generator reconstruct the pathological changes.
* Adjust the weights of Generator losses adaptively for balance of training.
* To start and stop the Standard discriminator and Task-induced discriminator at the right time according to experrience (see source code).
* The generalization performance of pre-trained Task-induced discriminatora is vitally important.

### Stage 2: PT-DCN for disease classification 
<p align="center">
  <img src="https://github.com/xiaoxingxingkz/TPA-GAN/blob/main/Figure_in_paper/Fig3.png" width="700">
</p>


Run the following code，we test the models after each epoch of training process. No independent test program is required, we also provide。
```python
train_PT_DCN.py
```
## About Data
Next project, We will provide clear pre-processing procedures of the neuroimages from ADNI database, and release our pre-processed data.
