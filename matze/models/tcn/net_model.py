import numpy as np
import torch as th
import torch.nn as nn
import tcn
import configuration as cfg


class Model(nn.Module):
    """
    This class contains the TCN network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, _params, _tensors):

        super(Model, self).__init__()

        self.params = _params
        self.tensors = _tensors

        #
        # Set up the TCN model

        # Temporal convolution network
        self.tcn = tcn.TemporalConvNet(
            num_inputs=self.params.input_size,
            num_channels=self.params.tconv_num_channels,
            kernel_size=self.params.tconv_kernel_size,
            dropout=0.0
        ).to(device=self.params.device)

    def forward(self, net_in):
        """
        Runs the forward pass of the CNN network for a given input
        :param net_in: The input for the network
        """

        # Convert the net_in numpy array to a tensor to feed it to the network
        net_in_tensor = th.from_numpy(net_in).to(device=self.params.device)

        # Forward the input through the conv1 layer
        tconv_out = self.tcn(net_in_tensor)

        # Update the output and hidden state tensors of the network
        self.tensors.output = tconv_out

    def reset(self, batch_size):
        self.tensors.reset(batch_size=batch_size)
