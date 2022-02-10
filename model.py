import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43  # GTSRB as 43 classes


class TrafficSignNet(nn.Module):
    """
    This is a class instance for creating a traffic sign network model
    """

    def __init__(self):
        """
        Constructor of Traffic sign network class
        It initializes the layer attributes of the Traffic sign network class

        Parameters
        ---------
        stn : Spatial Transformer Network layer
        conv1 : Convolutional layer 1 with input_channel =1, output_channel =100 and kernel_size =5
        pool : 2D Max pooling layer
        conv2 : Convolutional layer 2 with input_channel =100, output_channel =150 and kernel_size =3
        conv2_bn : Batch normalization layer with number of features =150
        conv3 : Convolutional layer 1 with input_channel =150, output_channel =250 and kernel_size =1
        conv3_bn : Batch normalization layer for the output from conv3
        fc1 : Fully connected layer with input features =250 * 3 * 3, output feature =350
        fc1_bn : Batch normalization layer with number of features =350
        fc2 : Fully connected layer with input features =350, output feature =43 (no. of classes)
        dropout: Dropout layer with dropout rate =0.5
        """
        super(TrafficSignNet, self).__init__()
        # TODO: Comment the below line to create a model without STN layer
        self.stn = Stn()
        # With square kernels and default stride
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        # pool of square window of size=2 and stride =2
        self.pool = nn.MaxPool2d(2, 2)
        # With square kernels
        self.conv2 = nn.Conv2d(100, 150, 3)
        # With Learnable Parameters
        self.conv2_bn = nn.BatchNorm2d(150)
        # With square kernels
        self.conv3 = nn.Conv2d(150, 250, 1)
        # With Learnable Parameters
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        # With Learnable Parameters
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        This method is the actual network transformation
        This method maps the input tensor to a prediction output tensor
        """
        # TODO: Comment the below line to build a model without STN layer
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return x


# TODO: Comment the class Stn if STH layer is removed in the model

class Stn(nn.Module):
    """
    This is a class instance to create a spatial transformer network
    """

    def __init__(self):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Linear(100, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


