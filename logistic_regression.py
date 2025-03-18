"""Working with Images and Logistic Regression""" 
import torch 
import torch.nn as nn 
from torch.utils.data import random_split 
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
#print(len(train_ds), len(val_ds)) 

"""It's essential to choose a random sample for creating a validation set. Training data is 
often sorted by the target labels. 

We can now create data loaders to help us load the data into batches. We'll use a batch size of 128"""
BATCH_SIZE = 128 
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE) 

"""We set shuffle=True for the training data loader to ensure that the batches generated in each epoch are different. This 
randomization helps generalize & speed up the training process. On the other hand, since the valiadation data loader
is used only for evaluating the model, there is no need to shuffle the images."""

"""MODEL
Now that we have prepared our data loaders, we can define our model. 
*  A logistic regression model is almost identical to a linear regresion model. It contains 
weights and bias matrices, and the output is obtained using simple matrix operations
(pred = x @ w.t() + b). 
* As we did with linear regression, we can use nn.Linear to create the model instead of manually
creating and initalizing the matrices. 
* Since nn.Linear expects each training example to be vectors, each 1x28x28 image tensor is flattened into a vector of size 784 (28x28) before being pased into the model.
* The output for each image is a vector of size 10, with each element signifying the probability of a particular label (0 to 9). The predicted label for an image is simply the one 
with the highest probability.""" 
input_size = 28*28 
num_classes = 10  

# Logistic regression model 
model = nn.Linear(input_size, num_classes) 

"""Of course, our model is a lot larger than  our previous model in terms of the number of parameters. Let's take a look at 
the weights and biases"""
#print(model.weight.shape)
#print(model.bias.shape) 

"""Although there are a total of 7850 parameters here, conceptually, nothing has changed so far. Let's try and generate some outputs using our model. We'll 
take the first 100 images from our dataset and pass them into our model"""

#for images, labels in train_dl:
#   print(labels)
#   print(images.shape)
#   outputs = model(images)  
#   break


"""The code above leads to an eror because our input data does not have the right shape. Our images are of the dhape 1x28x28, but we 
need them to be of vectors of size 784, i.e., we need to flatten them. We'll use the .reshape method of a tensor, which 
will allow us to efficently view each image as a flat vector without really creating a copy of the underlying data. To include this additional funcionality within our model, we need to define a custom model by extending nn.Module class from PyTorch.

A class in Python provides a "blueprint" for creating objects. Let's look at an example of defining a new class in Python."""

class Person: 
    # class constructor 
    def __init__(self, name, age):
        self.name = name 
        self.age = age  
    # Method 
    def say_hello(self): 
        print(f"Hello my name is {self.name}!") 

"""Here's how we create or instantiate an object of the class Person""" 
alex = Person("Alex", 25) 
#alex.say_hello()

"""Classes can also be build upon or textend the functionality of existing clases. Let's extend
the nn.Module class from PyTorch to define a custom model""" 

class MnistModel(nn.Module): 
    def __init__(self,):
        super().__init__() 
        self.linear = nn.Linear(input_size, num_classes) 

    def forward(self, xb) -> torch.Tensor: 
        xb = xb.reshape(-1, 784) 
        out = self.linear(xb) 
        return out  

model = MnistModel() 

for xb, yb in train_dl: 
    print(xb.shape)
    outputs = model(xb) 
    print(f"Outputs shape: {outputs.shape}") 
    print(f"Sample outputs: \n{outputs[:2].data}") 

    # Apply softmax for each output row 
    probs = F.softmax(outputs, dim=1) 

    # Look at sample probabilities 
    print(f"Sample probabilities: \n{probs[:2]}")

    # Add up the probabilties of an output row 
    print(f"Sum: {torch.sum(probs[0]).item()}")

    break  

"""To convert the output rows into proberbilities, we use the softmax function. 
While it's easy to implement the softmax function, we'll use the implementation that's provided withing PyTorch because it works well with 
multidimensional tensors (a list of output rows in our case)""" 