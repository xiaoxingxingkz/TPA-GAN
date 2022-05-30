# Task-Induced Pyramid and Attention GAN for Multimodal Brain Image Imputation and Classification in Alzheimer’s Disease
Paper link: https://ieeexplore.ieee.org/document/9490307

## Summary

In practice, multimodal images may be incomplete since PET is often missing due to high financial costs or availability. Most of the existing methods simply excluded subjects with missing data, which unfortunately reduced the sample size. To address these problems, we propose a task-induced pyramid and attention generative adversarial network (TPA-GAN) for imputation of multimodal brain images.

## Overview
![Fig1.png](https://github.com/xiaoxingxingkz/TPA-GAN/blob/main/Figure_in_paper/Fig1.png)

Fig. 1. Overview of our proposed deep learning framework: Stage1: Task-induced pyramid and attention GAN (TPA-GAN) for multimodal brain image imputation and, Stage2: Pathwise transfer dense convolution network (PT-DCN) for disease classification.


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

