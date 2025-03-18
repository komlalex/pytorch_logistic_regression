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
import pandas as pd 

from pathlib import Path
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

# Define accuracy function 
def accuracy(outputs, y_true): 
    y_preds = torch.argmax(outputs, dim=1)
    return torch.tensor(torch.sum(y_preds==y_true).item() / len(y_preds))

# Define loss function 
loss_fn = F.cross_entropy

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
    #print(xb.shape)
    outputs = model(xb) 
    #print(f"Outputs shape: {outputs.shape}") 
    #print(f"Sample outputs: \n{outputs[:2].data}") 

    # Apply softmax for each output row 
    probs = F.softmax(outputs, dim=1) 

    # Look at sample probabilities 
    #print(f"Sample probabilities: \n{probs[:2]}")

    # Add up the probabilties of an output row 
    #print(f"Sum: {torch.sum(probs[0]).item()}") 

    # Get prediction labels from probalities 
    preds = torch.argmax(probs, dim=1)  
    #print(preds) 

    # Compare predictions to actual labels
    df = pd.DataFrame({"y_true": yb, "y_preds": preds}) 
    #print(df) 

    # Calculate accuracy 
    acc = accuracy(outputs, yb) 
    #print(f"Accuracy: {acc: .4f}") 

    # Calculate loss 
    loss = loss_fn(outputs, yb) 
    #print(f"Loss: {loss: .5f}")

    break  

"""To convert the output rows into proberbilities, we use the softmax function. 
While it's easy to implement the softmax function, we'll use the implementation that's provided withing PyTorch because it works well with 
multidimensional tensors (a list of output rows in our case)""" 

"""Finally, we determined the predicted label for each image by simply choosing the index of the highest probability in each output row. We can do this by using the torch.armax, the index of the rows's largest
element OR torch.max which returns each row's largest element and the corresponding index""" 

"""Evaluation Metrics and Loss Function 
Just as linear regression, we need a way to evaluate how well our model is performing. A 
natural way to do this would be to find the percentage of labels that were predicted correctly
"""

"""Accuracy si not an excellent way for us(humans) to evaluate the model. However, it can't be used as a loss function for optimizing our model using 
gradient descent for the following reasons: 
1. It's not a differentiable function. torch.argmax or torch.max and == are both contiguous and non-diferentiable operations, 
se we can't use acccuracy for computing gradients w.r.t the weights and biases
2. It doesn't take into account the actual probabilities predicted by the model, so it can't provide
sufficient feedback for incremental improvements 

For these reasons, accuracy is often used as an evaluation metric for clasification, but not as a loss function. 
A commonly used loss function for classification problems is cross-entropy. 

While it may seem complicated, it's quite simple: 
* For each output row, pick the predicted probability for correct label 
* Then take the logarithm of the picked probability. If the probability is high, i.e. close to 1, 
then its logaritm is a very small negative close to 0. And if the probability is low (close to 0), then its 
logarithm is a very big negative value. We must multiply the result by -1, which results in a large positve value 
of loss for poor predictions
* Finally, the average of the cross entropy across all the output rows to get the overall loss for a batch of data. 

Unlike accuracy, cross-entropy is a contiguous and differentiable function. It also provides useful fedback for incremental improvents in the model (slightly higher proberbility for correct label leads to a lower loss). 
These two factors make cross-entropy a better choice for the loss function. 

As you might expect, PyTorch provides an efficient implementation of cross-entropy as part of the torch.nn.funcrional package. 
Moreover, it also performs softmax internally, so we can directly pass the model's predictions wihtout converting into probabilities. 

We know that cross entropy is the negative logarithm of the predicted probability of the correct label averaged over all training samples. Therefore, one way to interpret the resulting number e.g. 2.23 is look at 
e^-2.23 which is around 0.1 as the predicted probability of the correct label, on average. The lower the loss, the better 
the model."""  

"""TRAINING THE MODEL 
Now that we defined the data loaders, model, loss function and optimizer, we are ready to train the model. The training process is 
identical to linear regression, with the addition of "validation phase" to evalaute the model in each epoch. 
Here's what it looks like in pseudocode:

for epoch in range(num_epochs): 
    # Training phase 
    for batch in train_dl: 
        # Generate predictions 
        # Calculate loss 
        # Compute gradients 
        # update weights 
        # Reset gradients 
    
    # Validation phase 
    for batch in val_dl: 
        # Generate predictions 
        # Calculate loss 
        # Calculate metrics (accuracy, etc.)
    # Calculate average validation loss and metrics 
    # Log epoch, loss and metrics for inspection 

Some parts of the training loop are specific to the problem we're solving (e.g loss, metrics) whereas 
others are generic and can be applied to any deep learning problem. 

We'll include the problem-independent parts within a function called fit, which will be used 
to train the model. The problem-specific parts will be implemented by adding new methods to the nn.Module class
"""""

def fit(epochs: int, lr: float, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, opt_func=torch.optim.SGD): 
    optimizer = opt_func(model.parameters(), lr) 
    history = [] # recording epoch-wise results

    for epoch in range(epochs): 

        # Training Phase
        for batch in train_dl: 
            loss = model.training_step(batch) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

        # Validation phase 
        result = evaluate(model, val_dl)
        model.epoch_end(epoch, result) 
        history.append(result) 
    return history
            
def evaluate(model, val_dl): 
    outputs = [model.validation_step(batch) for batch in val_dl] 
    return model.validation_epoch_end(outputs) 

"""Finally let's redefine the MnistModel class to include aditional methods training_step, 
validation_step, validation_epoch_end, and epoch_end used by fit and evalaute"""

class MnistModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb): 
        xb = xb.reshape(-1, 784)
        out = self.linear(xb) 
        return out 
    
    def training_step(self, batch): 
        images, labels = batch 
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss 
        return loss 
    
    def validation_step(self, batch): 
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels)         # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc} 
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs] 
        epoch_loss = torch.stack(batch_losses).mean() # Combine losses and find average
        batch_accs = [x["val_acc"] for x in outputs] 
        epoch_acc = torch.stack(batch_accs).mean()    # Combine accuracies and find average
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, result): 
        print(f"\33[32m Epoch: {epoch + 1} val_loss: {result["val_loss"]:.4f} val_acc: {result["val_acc"]:.4f}") 

model = MnistModel()


"""Before we train the model, let's see how the model performs on the validation set 
with the initial set of randomly initialized weights & biases"""

result0 = evaluate(model, val_dl)
#print(result0)

"""The intial accuracy is around 19%, which one might expect from a randomly intialized 
model (since it has a 1 in 10 chance of getting a label right by guessing)"""

history = fit(5, 0.03, model, train_dl, val_dl) 

"""That's a great result as our model has reached an accuracy of around 90%. 
While the accuracy does continue to increase as we train for more epochs, the improvement gets smaller 
with every epoch. Let's visualize this using a line graph""" 

accuracies = [result["val_acc"] for result in history] 
plt.figure(figsize=(10, 9))
plt.plot(accuracies, "-x")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy vs No. of epochs") 
#plt.show()


"""Testing with Individual Images 

While we have been tracking the overall accuracy of a model so far, it's also a good idea to look at the 
model's results on some sample images. Let's test our model with some images from the predefined test dataset of 10,000
images. We begin by recreating the test dataset with the ToTensor transform
""" 
# Define test dataset 
test_dataset = MNIST(root="data/", 
                     train=False, 
                     download=True, 
                     transform=transforms.ToTensor()) 

"""Here's a sample image from the dataset"""
img, label = test_dataset[0] 
plt.figure(figsize=(10, 9))
plt.imshow(img[0], cmap="gray")
plt.title(f"Label: {label}") 

"""Let's write a helper function predict_image, which return the predicted label for a single image tensor"""
def predict_image(img, model): 
    xb = img.unsqueeze(0)
    yb = model(xb) 
    preds = torch.argmax(yb, dim=1) 
    return preds[0].item()  

img, label = test_dataset[0] 
plt.figure(figsize=(10, 10))
plt.imshow(img[0], cmap="gray") 
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}") 

img, label = test_dataset[10] 
plt.figure(figsize=(10, 10))
plt.imshow(img[0], cmap="gray") 
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}")  

img, label = test_dataset[193] 
plt.figure(figsize=(10, 10))
plt.imshow(img[0], cmap="gray") 
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}") 

img, label = test_dataset[1839] 
plt.figure(figsize=(10, 10))
plt.imshow(img[0], cmap="gray") 
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}") 

#plt.show()

"""
Identifying where our model performs poorly can help us improve the model, by collecting more training data, 
increasing/decreasing the complexity of the model, changing the hyperparameters. 
As a final step, let's also look at the overall loss and accuracy of the model on the test set.
"""
test_dl = DataLoader(test_dataset, batch_size=256) 
result = evaluate(model, test_dl) 
print(result)

"""We expect this to be similar to the accuracy/loss on the validation set. If not, we might need a better validation set that has similar data 
and distribution as the test set (which often comes from real world data)"""

"""SAVING AND LOADING THE MODEL 
Since we've trained our model for a long time and achieved a reasonable accuracy, it would be a good idea to save the weights and 
bias matrices to disk, so that we can reuse the model later to avoid retraining from scratch. Here's how to save the model"""
MODELS_PATH = Path("models/")
MODELS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODELS_PATH / "mnist-logistic.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH) 