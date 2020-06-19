## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
#from collections import OrderedDict

IMAGE_SIZE = 224     # square image: W == H
OUTPUT_SIZE = 136    # 68 keypoints


class Net12(nn.Module):

    IN_CHANNELS_CONV1 = 1    # grayscale image
    OUT_CHANNELS_CONV1 = 16
    KERNEL_SIZE_CONV1 = 7    # square kernel
    STRIDE_CONV1 = 3         # reduce image size faster
    KERNEL_SIZE_POOL = 3     # default stride: equal to kernel size
    OUT_FEATURES_FC1 = 1000


    def __init__(self):
        super(Net12, self).__init__()

        # calculation out width of output map
        # general case described in https://pytorch.org/docs/stable/nn.html#conv2d
        # here only specific case: (W-F) // S + 1
        out_size_conv1 = (IMAGE_SIZE - self.KERNEL_SIZE_CONV1)//self.STRIDE_CONV1 + 1
        self.conv1 = nn.Conv2d(self.IN_CHANNELS_CONV1, self.OUT_CHANNELS_CONV1, 
                               self.KERNEL_SIZE_CONV1, stride=self.STRIDE_CONV1)
                 
        # calculation out width of output map after maxpool
        # general case described in https://pytorch.org/docs/stable/nn.html#maxpool2d
        # here only specific case: (W-F) // S + 1
        out_size_pool = out_size_conv1 // self.KERNEL_SIZE_POOL
        self.pool = nn.MaxPool2d(self.KERNEL_SIZE_POOL)  

        # calculate number of input features for fully connected layers:
        # number of channels * number of pixels in output from pooling layer
        self.in_features_fc1 = self.OUT_CHANNELS_CONV1 * out_size_pool**2  # square
        self.fc1 = nn.Linear(self.in_features_fc1, self.OUT_FEATURES_FC1)
        self.fc2 = nn.Linear(self.OUT_FEATURES_FC1, 136)

        # TODO: Weight initialisation


        
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)          
        x = x.view(-1, self.in_features_fc1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        
        return x


class Net1(nn.Module):
    
    def __init__(self):
        super(Net1, self).__init__()
        # input size: 224 x 224
        # output size = (W-F)/S +1 = (224-7)/3 + 1 = 73
        # after pooling: 24
        self.conv1 = nn.Conv2d(1, 16, 7, stride=3)
                 
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout 
        # or batch normalization) to avoid overfitting
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(3, 3)  
        self.fc1 = nn.Linear(16*24*24, 1000)
        self.fc2 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.pool(x)          
        x = x.view(-1, 16*24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        

        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net2(nn.Module):
    
    def __init__(self):
        super(Net2, self).__init__()
        # input size: 224 x 224
        # output size = (W-F)/S +1 = (224-5)/2 + 1 = 110
        # after pooling: 55
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        # input size: 55 x 55
        # output size = (W-F)/S +1 = (55-3)/1 + 1 = 53
        # after pooling: 26
        self.conv2 = nn.Conv2d(16, 32, 3)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout 
        # or batch normalization) to avoid overfitting
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)  
        self.fc1 = nn.Linear(32*26*26, 6000)
        self.fc2 = nn.Linear(6000, 1000)
        self.fc3 = nn.Linear(1000, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = self.pool(x)          
        x = F.relu(self.conv2(x))
        x = self.pool(x)          
        x = x.view(-1, 32*26*26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        

        # a modified x, having gone through all the layers of your model, should be returned
        return x
    

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
