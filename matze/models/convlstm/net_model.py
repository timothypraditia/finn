import numpy as np
import torch as th
import torch.nn as nn
import conv_lstm
import configuration as cfg


class Model(nn.Module):
    """
    This class contains the CNN network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, _params, _tensors):

        super(Model, self).__init__()

        self.params = _params
        self.tensors = _tensors

        #
        # Set up the convLSTM model

        # First convolution layer (16x16 -> 16x16)
        self.convlstm1_layer = conv_lstm.ConvLSTMCell(
            input_size=self.params.convlstm1_input_size,
            input_channels=self.params.convlstm1_input_channels,
            hidden_channels=self.params.convlstm1_hidden_channels,
            kernel_size=self.params.convlstm1_kernel_size,
            bias=self.params.convlstm1_bias
        ).to(device=self.params.device)

        # Second convolution layer (16x16 -> 16x16)
        self.convlstm2_layer = conv_lstm.ConvLSTMCell(
            input_size=self.params.convlstm2_input_size,
            input_channels=self.params.convlstm2_input_channels,
            hidden_channels=self.params.convlstm2_hidden_channels,
            kernel_size=self.params.convlstm2_kernel_size,
            bias=self.params.convlstm2_bias
        ).to(device=self.params.device)

    def forward(self, net_in):
        """
        Runs the forward pass of the CNN network for a given input
        :param net_in: The input for the network
        """

        # Convert the net_in numpy array to a tensor to feed it to the network
        net_in_tensor = th.from_numpy(net_in).to(device=self.params.device)

        # Forward the input through the convlstm1 layer
        convlstm1_out, convlstm1_c = self.convlstm1_layer(
            input_tensor=net_in_tensor,
            cur_state=(self.tensors.convlstm1_h, self.tensors.convlstm1_c)
        )

        # Forward the input through the convlstm2 layer
        convlstm2_out, convlstm2_c = self.convlstm2_layer(
            input_tensor=convlstm1_out,
            cur_state=(self.tensors.convlstm2_h, self.tensors.convlstm2_c)
        )

        # Update the output and hidden state tensors of the network
        self.tensors.output = convlstm2_out
        self.tensors.convlstm1_h = convlstm1_out
        self.tensors.convlstm1_c = convlstm1_c
        self.tensors.convlstm2_h = convlstm2_out
        self.tensors.convlstm2_c = convlstm2_c

    def reset(self, batches):
        self.tensors.reset(batches=batches)
