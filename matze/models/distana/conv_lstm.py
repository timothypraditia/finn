import torch as th
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM implementation using Conv1d instead of Conv2d operations.
    """

    def __init__(self, batch_size, input_size, lstm_cells, field_size, device,
                 bias=True):
        super(ConvLSTMCell, self).__init__()

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.lstm_cells = lstm_cells
        self.field_size = field_size
        self.bias = bias

        # Hidden (h) and cell (c) states
        self.h = th.zeros(size=(batch_size, lstm_cells, field_size),
                          device=device)
        self.c = th.zeros(size=(batch_size, lstm_cells, field_size),
                          device=device)

        self.conv = nn.Conv1d(in_channels=input_size + lstm_cells,
                              out_channels=lstm_cells*4,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=bias)

    def reset_states(self):
        self.h = th.zeros_like(self.h)
        self.c = th.zeros_like(self.c)

    def reset_parameters(self):
        # Uniform distribution initialization of lstm weights with respect to
        # the number of lstm cells in the layer
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h_prev=None, c_prev=None):
        
        # Set the previous hidden and cell states if not provided
        h_prev = self.h if h_prev is None else h_prev
        c_prev = self.c if c_prev is None else c_prev

        # Perform input and recurrent convolutions
        conv_res = self.conv(th.cat((x, h_prev), dim=1))

        # combined_conv = th.cat((input_conv, recur_conv), dim=1)
        netin, igate, fgate, ogate = th.split(conv_res, self.lstm_cells,
                                              dim=1)

        # Compute input and gate activations
        act_input = th.tanh(netin)
        act_igate = th.sigmoid(igate)
        act_fgate = th.sigmoid(fgate)
        act_ogate = th.sigmoid(ogate)

        # Compute the new cell and hidden states
        c_curr = act_fgate * c_prev + act_igate * act_input
        h_curr = act_ogate * th.tanh(c_curr)

        # Update the hidden and cells states
        self.h = h_curr
        self.c = c_curr

        return h_curr, c_curr