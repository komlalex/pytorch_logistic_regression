"""Working with Images and Logistic Regression""" 
import torch 
import torch.nn as nn 

import torchvision 
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt  

# Download training dataset 
dataset = MNIST(root="data/", download=True) 

"""The dataset has 60000 images that we'll use to train the model. There is an additional 10,0000 images
used for evaluating models and reporting metrics in paper reports. We can create the test dataset using the MNIST class by passing train=False to the 
constructor"""
test_dataset = MNIST(root="data/", train=False, download=True) 
print(len(test_dataset))
