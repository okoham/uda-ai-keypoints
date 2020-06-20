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


def conv_dims(input_size, k, s, p):
    """Return output size for a 2D convolution kernel (square input, 
    square kernel). See https://arxiv.org/pdf/1603.07285v2.pdf, relationship 6

    Parameters
    ----------
    input_size : int
        size (width) of input feature
    k : int
        convolution kernel size
    s : int
        convolution stride
    p : int 
        zero padding

    Returns
    -------
    output_size : int
        size (width) of output feature
    
    """
    return (input_size + 2*p - k)//s + 1



def pool_dims(input_size, k, s):
    """Return output size for a 2D convolution kernel (square input, 
    square kernel). See https://arxiv.org/pdf/1603.07285v2.pdf, relationship 7

    Parameters
    ----------
    input_size : int
        size (width) of input feature
    k : int
        pooling kernel size
    s : int
        pooling stride

    Returns
    -------
    output_size : int
        size (width) of output feature
    
    """
    return (input_size - k)//s + 1


class Net12(nn.Module):

    IN_CHANNELS_CONV1 = 1    # grayscale image
    OUT_CHANNELS_CONV1 = 16
    KERNEL_SIZE_CONV1 = 7    # square kernel
    STRIDE_CONV1 = 3         # reduce image size faster
    KERNEL_SIZE_POOL = 3     # default stride: equal to kernel size
    OUT_FEATURES_FC1 = 1000


    def __init__(self):
        super().__init__()

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
        self.fc2 = nn.Linear(self.OUT_FEATURES_FC1, OUTPUT_SIZE)

        # TODO: Weight initialisation


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)          
        x = x.view(-1, self.in_features_fc1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        
        return x



class Net12d(Net12):

    DROP1 = 0.5
    DROP2 = 0.5

    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout(self.DROP1)
        self.drop2 = nn.Dropout(self.DROP2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(-1, self.in_features_fc1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x



class Net22(nn.Module):

    def __init__(self):
        super().__init__()

        # first convolution
        i1, m1, n1 = IMAGE_SIZE, 1, 16    # input size, num features in, num out
        k1, s1, p1 = 5, 2, 0              # kernel size, stride, padding
        o1 = conv_dims(i1, k1, s1, p1)    # output size
        # first pooling
        kp1, sp1 = 2, 2                   # kernel size, stride  
        op1 = pool_dims(o1, kp1, sp1)     # output size
        # second convolution
        i2, m2, n2 = op1, n1, 32          # input size, num features in, num out
        k2, s2, p2 = 3, 1, 0              # kernel size, stride, padding
        o2 = conv_dims(i2, k2, s2, p2)    # output size
        # second pooling
        kp2, sp2 = 2, 2                   # kernel size, stride
        op2 = pool_dims(o2, kp2, sp2)     # output size

        self.conv1 = nn.Conv2d(m1, n1, k1, stride=s1, padding=p1, bias=False)
        self.conv2 = nn.Conv2d(m2, n2, k2, stride=s2, padding=p2, bias=False)
        self.pool1 = nn.MaxPool2d(kp1, stride=sp1)
        self.pool2 = nn.MaxPool2d(kp2, stride=sp2)

        # fully connected layers
        self.mh0 = n2 * op2**2            # needed later - size of first linear layer
        mh1 = 2000                        # input size of second fully connected layer

        self.fc1 = nn.Linear(self.mh0, mh1)
        self.fc2 = nn.Linear(mh1, OUTPUT_SIZE)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.mh0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Net22d(Net22):

    def __init__(self):
        super().__init__()
        self.drop1 = nn.Dropout2d(0.25)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        x = x.view(-1, self.mh0)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)

        return x



class Net32d(nn.Module):

    def __init__(self):
        super().__init__()

        # first convolution
        i1, m1, n1 = IMAGE_SIZE, 1, 16    # input size, num features in, num out
        k1, s1, p1 = 3, 1, 0              # kernel size, stride, padding
        o1 = conv_dims(i1, k1, s1, p1)    # output size
        # first pooling
        kp1, sp1 = 2, 2                   # kernel size, stride
        op1 = pool_dims(o1, kp1, sp1)     # output size
        # second convolution
        i2, m2, n2 = op1, n1, 32          # input size, num features in, num out
        k2, s2, p2 = 3, 1, 0              # kernel size, stride, padding
        o2 = conv_dims(i2, k2, s2, p2)    # output size
        # second pooling
        kp2, sp2 = 2, 2                   # kernel size, stride
        op2 = pool_dims(o2, kp2, sp2)     # output size
        # third convolution
        i3, m3, n3 = op2, n2, 64          # input size, num features in, num out
        k3, s3, p3 = 3, 1, 0              # kernel size, stride, padding
        o3 = conv_dims(i3, k3, s3, p3)    # output size
        # third pooling
        kp3, sp3 = 2, 2                   # kernel size, stride
        op3 = pool_dims(o3, kp3, sp3)     # output size

        self.conv1 = nn.Conv2d(m1, n1, k1, stride=s1, padding=p1, bias=False)
        self.conv2 = nn.Conv2d(m2, n2, k2, stride=s2, padding=p2, bias=False)
        self.conv3 = nn.Conv2d(m3, n3, k3, stride=s3, padding=p3, bias=False)
        self.pool1 = nn.MaxPool2d(kp1, stride=sp1)
        self.pool2 = nn.MaxPool2d(kp2, stride=sp2)
        self.pool3 = nn.MaxPool2d(kp3, stride=sp3)

        # fully connected layers
        self.mh0 = n3 * op3**2            # needed later - size of first linear layer
        mh1 = 2000                        # input size of second fully connected layer

        self.fc1 = nn.Linear(self.mh0, mh1)
        self.fc2 = nn.Linear(mh1, OUTPUT_SIZE)

        self.drop2 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, self.mh0)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)

        return x


class Net33d(nn.Module):

    def __init__(self):
        super().__init__()

        # first convolution
        i1, m1, n1 = IMAGE_SIZE, 1, 16    # input size, num features in, num out
        k1, s1, p1 = 4, 1, 0              # kernel size, stride, padding
        o1 = conv_dims(i1, k1, s1, p1)    # output size
        # first pooling
        kp1, sp1 = 2, 2                   # kernel size, stride
        op1 = pool_dims(o1, kp1, sp1)     # output size
        # second convolution
        i2, m2, n2 = op1, n1, 32          # input size, num features in, num out
        k2, s2, p2 = 3, 1, 0              # kernel size, stride, padding
        o2 = conv_dims(i2, k2, s2, p2)    # output size
        # second pooling
        kp2, sp2 = 2, 2                   # kernel size, stride
        op2 = pool_dims(o2, kp2, sp2)     # output size
        # third convolution
        i3, m3, n3 = op2, n2, 64          # input size, num features in, num out
        k3, s3, p3 = 2, 1, 0              # kernel size, stride, padding
        o3 = conv_dims(i3, k3, s3, p3)    # output size
        # third pooling
        kp3, sp3 = 2, 2                   # kernel size, stride
        op3 = pool_dims(o3, kp3, sp3)     # output size

        self.conv1 = nn.Conv2d(m1, n1, k1, stride=s1, padding=p1, bias=False)
        self.conv2 = nn.Conv2d(m2, n2, k2, stride=s2, padding=p2, bias=False)
        self.conv3 = nn.Conv2d(m3, n3, k3, stride=s3, padding=p3, bias=False)
        self.pool1 = nn.MaxPool2d(kp1, stride=sp1)
        self.pool2 = nn.MaxPool2d(kp2, stride=sp2)
        self.pool3 = nn.MaxPool2d(kp3, stride=sp3)

        # fully connected layers
        self.mh0 = n3 * op3**2            # needed later - size of first linear layer
        mh1, mh2 = 2000, 2000             # input size of second fully connected layer

        self.fc1 = nn.Linear(self.mh0, mh1)
        self.fc2 = nn.Linear(mh1, mh2)
        self.fc3 = nn.Linear(mh2, OUTPUT_SIZE)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.view(-1, self.mh0)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = F.relu(self.fc2(x))
        x = self.drop5(x)
        x = self.fc3(x)

        return x