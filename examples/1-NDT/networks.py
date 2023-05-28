import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

class DenseNet(nn.Module):
    """
    This is a DenseNet Class.
    -> layersizes: number of neurons in each layer. 
                    E.g. [dim_in, 32, 32, 32, dim_out]
                    where, dim_in and dim_out are network's input and output dimension respectively
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, layersizes=[2, 32, 32, 32, 1], activation=nn.Sigmoid()):
        super(DenseNet, self).__init__()
        
        self.layersizes = layersizes
        self.activation = activation
        
        self.input_dim,  self.hidden_sizes, self.output_dim = self.layersizes[0], self.layersizes[1:-1], self.layersizes[-1]
        
        self.nlayers = len(self.hidden_sizes) + 1
        self.layers = nn.ModuleList([])
        for i in range(self.nlayers):
            self.layers.append( nn.Linear(self.layersizes[i], self.layersizes[i+1]) )


    def forward(self, x):
        
        for i in range(self.nlayers-1):
            x = self.activation(self.layers[i](x))
         
        # no activation for last layer
        out = self.layers[-1](x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)  


class DenseResNet(nn.Module):
    """
    This is a DenseResNet Class.
    -> dim_in: network's input dimension
    -> dim_out: network's output dimension
    -> num_resnet_blocks: number of ResNet blocks
    -> num_layers_per_block: number of layers per ResNet block
    -> num_neurons: number of neurons in each layer
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid()):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation

        self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])

        self.last = nn.Linear(num_neurons, dim_out)

    def forward(self, x):

        x = self.activation(self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.resblocks[i][j](z))

            x = z + x

        out = self.last(x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)  