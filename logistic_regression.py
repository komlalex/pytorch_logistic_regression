"""Working with Images and Logistic Regression""" 
import torch 
import torch.nn as nn 
from torch.utils.data import random_split

import torchvision 
from torchvision.datasets import MNIST 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  

# Download training dataset 
dataset = MNIST(root="data/", download=True) 

"""The dataset has 60000 images that we'll use to train the model. There is an additional 10,0000 images
used for evaluating models and reporting metrics in paper reports. We can create the test dataset using the MNIST class by passing train=False to the 
constructor"""
test_dataset = MNIST(root="data/", train=False, download=True) 

#print(dataset[0])

"""The dataset is a pair consisting of a 28x28 image and a label. The image is an object of class PIL.Image.Image, which is a
part of Python imaging library Pillow. We can view the image using matplotlib, the 
de-facto plotting and graphing library for data science in Python.""" 


plt.figure(figsize=(10,8))
image, label = dataset[0] 
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray") 
plt.title(f"Label: {label}")
plt.axis(False) 

image, label = dataset[10] 
plt.subplot(1, 2, 2)
plt.imshow(image, cmap="gray") 
plt.title(f"Label: {label}")
plt.axis(False)
#plt.show()  

"""It's evident that these images are relatively small in size, and recognizing the digits can sometimes
be challenging even for the human eye. While it's useful to look at these images, there's just one problem here: 
PyTorch doesn't know how to work with images. We need to convert the images into tensors. We can do this by 
specifying a transform while creating our dataset. 

PyTorch datasets allow us to specify one or more transformation functions that are applied to the images as they
are loaded. The torchvision.transforms module contains many such predefined functions.
"""
dataset = MNIST(root="data/", 
                download=True, 
                transform=transforms.ToTensor()) 

test_dataset = MNIST(root="data/", train=False, 
                     download=True, 
                     transform=transforms.ToTensor()) 

img_tensor, label = dataset[0] 
#print(img_tensor.shape, label) 
#print(img_tensor[0, 10:15, 10:15]) 

"""The values range from 0 to 1, with 0 representing black and one representing white, and the values
in between different shades of rey. We can also plot the tensor as an image using plt.show"""

# Plot the image by passing in the 28x28 matrix 
plt.figure()
plt.imshow(img_tensor[0, 10:15, 10:15], cmap="gray")
#plt.show() 

"""TRAINING AND VALIDATION DATASETS

While building real-world machine learning models, it is quite common to split the dataset into three: 
1. Training set - used to train the model, i.e., compute the loss and adjust the model's weights using 
gradient descent.
2. Validation set - used to evaluate the model during training, adjust hyperparameters (learning rate, etc), 
and pick the best version of the model
3. Test set - used to compare different models or approaches and report the model's 
final accuracy. 

In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different
researches can report their models' resultsss against the same collection of images.

Since there's no predefined validation set, we must manually split the 60,000 images into training and valiadation 
datasets. Let's set aside 10,000 randomly chosen images for validation. We can do this using the random_split method from PyTorch
"""
train_ds, val_ds = random_split(dataset, [50_000, 10_000]) 
print(len(train_ds), len(val_ds))