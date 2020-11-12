# pytorch tutorials
# from https://pytorch.org/tutorials/
# Initialized 11/03/2020

# %% modules
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# %%
# _____
# 1) 60 minute blitz
# _____
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html


# %%
#_____
# What is PyTorch?
# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
# ____

# uninitalized matrix
x = torch.empty(5, 3)
print(x)

# random matrix
x = torch.rand(5, 3)
print(x)

#zeros
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# directly from data list
x = torch.tensor([5.5, 3])
print(x)

# create from existing 
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)

# get size
print(x.size())

# add1
y = torch.rand(5, 3)
print(x + y)
# add2
print(torch.add(x, y))
# add output
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# add inplace
y.add_(x)
print(y)

# numpy-like indexing
print(x[:, 1])

# resize using torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# get the items in an array as a python number using .item()
# only does one at a time
x = torch.randn(1)
print(x)
print(x.item())

# convert to numpy arrays and back
a = torch.ones(5)
print(a)
# to numpy
b = a.numpy()
print(b)
# add to both
a.add_(1)
print(a)
print(b)

# convert to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# moving tensors onto 'device' using '.to'
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# %%
# Autograd: Automatic Differentiation
# The autograd package provides automatic
# differentiation for all operations on tensors
# a define-by-run framework
# backprop is defined by how your code is run
# every single iteration is different

# <torch.Tensor> is central to the package
# set attribute (<.requires_grad> as True) to track all operations
# <.backward()> after you finish your computation to have
# <.grad> attribute shows accumulated gradient
# <.detach()> stop a tensor from tracking computation history
# to prevent tracking history, wrap code block in
#   <with torch.no_grad()>
# <Tensor> and <Function> build an acyclic graph to 
#   encode complete history
# This is tracked using the <.grad_fn> attribute

# To compute derivatives, call <Tensor.backward()
# If Tensor is scalar, specify no args for <.backward()>

# %%
# Autograd A: Tensor
# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor operation:
y = x + 2
print(y)

# y was created as a result of an operation, so it has a grad_fn.
print(y.grad_fn)

# Do more operations on y
z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_( ... ) changes an existing 
#   Tensor’s requires_grad flag in-place. 
#   The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# %% 
# Autograd B: Gradients
# more reference:
# # https://pytorch.org/docs/stable/autograd.html#function

# backprop now
# assume 'out' contains a single scaler
# equivalent to out.backward(torch.tensor(1.)).
# commented out to prevent redundancy
out.backward()
print(x.grad)

# example vector Jacobian Product
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# if y is no longer a scale, torch.autograd cannot
#   run directly
#   so we ned to pass the vector to backward
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

# stop autograd from tracking history
# <.requires_grad=True>:
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# or use <.detach()> to get a new tensor without gradient
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

# %%
# # Neural Networks:
# use <torch.nn> package
# <nn> depends on <autograd> to define models
#   and differentiate them
# An <nn.Module> contains layers, and
#   a method <forward(input)> that returns <output)

# Typical Training Situation
# # 1) Define the neural network that has some learnable parameters (or weights)
# # 2) Iterate over a dataset of inputs
# # 3) Process input through the network
# # 4) Compute the loss (how far is the output from being correct)
# # 5) Propagate gradients back into the network’s parameters
# # 6) Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

# %%
# 1) Neural Networks: Define the network

# note that you only need to define the forward function, because backward is 
# done automatically using autograd

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# the learnable parameters are returned by <net.parameters()>
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# %%
# 2) Neural Networks: Add Inputs
# try random 32x32 input (note the network is expecting a 32x32 input)
# NOTE* Further Questions -> How do I know what input size is expected?
input = torch.randn(1, 1, 32, 32)

# %%
# 3) Neural Networks: Process input through Network
# NOTE* Still works with inputs of different dimensions
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backdrops (why?)
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
# 4) Neural Networks: Compute the loss
# # Many Loss Functions
# # https://pytorch.org/docs/stable/nn.html#loss-functions
# # simple is nn.MSELoss

output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()


loss = criterion(output, target)
print(loss)

# NOTE* What?
# follow a few steps back
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# %%
# 5) Neural Networks: Backpropogation
# To backpropagate the error all we have to do is to <loss.backward()>
# be sure to clear existing gradients
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# %%
# 6) Neural Networks: Update the weights in the network
# E.G. Stochastic Gradient Descent (SGD)
#   weight = weight - learning_rate * gradient

# simple implementation
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# complicated implementation
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

# %%
# Next time: 
# Training an Image Classifier: 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# with text, image, audiio, video data
#   load into numpy, convert into torch.*Tensor

# For this tutorial, we will use the CIFAR10 dataset. 
# It has the classes: ‘airplane’, ‘automobile’, 
# ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, 
# ‘ship’, ‘truck’. 
# The images in CIFAR-10 are of size 3x32x32, 
# i.e. 3-channel color images of 32x32 pixels 
# in size.

# Order: 
# 1 Load and normalizing the CIFAR10 training and test datasets using torchvision
# 2 Define a Convolutional Neural Network
# 3 Define a loss function
# 4 Train the network on the training data
# 5 Test the network on the test data
 
 # %%
# 1 Normalize CIFAR training and test dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# %%
# look at training images
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %%
# 2 Define a Convolutional Neural Network:
# NOTE* What's going on here?

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# %%
# 3 Define Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
# 4 Train the Network
# Loop over data iterator, and feed inputs to network and optimize
# LONG step


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# %%
# Test Network
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# %%
