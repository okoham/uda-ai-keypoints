## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # input size: 224 x 224
        # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        # after pooling: 110
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # input size: 110 x 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # after pooling: 54        
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # input size: 54 x 54
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # after pooling: 26        
        self.conv3 = nn.Conv2d(64, 128, 3)        
        
        # input size: 26 x 26
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # after pooling: 12        
        self.conv4 = nn.Conv2d(128, 256, 3)              
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout 
        # or batch normalization) to avoid overfitting
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(256*12*12, 6400)
        self.fc2 = nn.Linear(6400, 1000)
        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.pool(x)        
        x = F.relu(self.conv2(x))
        x = self.pool(x)        
        x = F.relu(self.conv3(x))
        x = self.pool(x)        
        x = F.relu(self.conv4(x))
        x = self.pool(x)     
        x = x.view(-1, 256*12*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        

        # a modified x, having gone through all the layers of your model, should be returned
        return x
