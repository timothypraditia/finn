import torch as th
import torch.nn as nn
import configuration as cfg
from conv_lstm import ConvLSTMCell


class DISTANA(nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, params, tensors):

        super(DISTANA, self).__init__()

        self.params = params
        self.tensors = tensors

        # Lateral input convolution layer
        self.lat_in_conv_layer = nn.Conv1d(
            in_channels=params.pk_lat_size,
            out_channels=params.pk_lat_size,
            kernel_size=params.pk_conv_ksize,
            stride=params.pk_conv_stride,
            padding=params.pk_conv_padding,
            bias=True
        ).to(device=self.params.device)

        # Dynamic and lateral input preprocessing layer
        self.pre_layer = nn.Conv1d(
            in_channels=params.pk_dyn_in_size + params.pk_lat_size,
            out_channels=params.pk_pre_layer_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        ).to(device=self.params.device)

        # Central LSTM layer
        self.clstm = ConvLSTMCell(
            batch_size=self.params.batch_size,
            input_size=params.pk_pre_layer_size,
            lstm_cells=params.pk_num_lstm_cells,
            field_size=params.field_size,
            device=params.device
        ).to(device=self.params.device)

        # Postprocessing layer
        self.post_layer = nn.Conv1d(
            in_channels=params.pk_num_lstm_cells,
            out_channels=params.pk_dyn_out_size + params.pk_lat_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        ).to(device=self.params.device)

    def forward(self, dyn_in, pk_stat_in=None, tk_stat_in=None):
        """
        Runs the forward pass of all PKs and TKs, respectively, in parallel for
        a given input
        :param dyn_in: The dynamic input for the PKs
        :param pk_stat_in: (optional) The static input for the PKs
        :param tk_stat_in: (optional) The static input for the TKs
        """

        # Write the dynamic PK input to the corresponding tensor
        if isinstance(dyn_in, th.Tensor):
            pk_dyn_in = dyn_in
        else:
            pk_dyn_in = th.from_numpy(
                dyn_in
            ).to(device=self.params.device)

        # Compute the lateral input as convolution of the lateral outputs from
        # the previous timestep
        pk_lat_in = self.lat_in_conv_layer(self.tensors.pk_lat_out.clone())

        # Forward the dynamic and lateral inputs through the preprocessing
        # layer
        pk_dynlat_in = th.cat(tensors=(pk_dyn_in, pk_lat_in), dim=1)
        pre_act = th.tanh(self.pre_layer(pk_dynlat_in))

        # Feed the preprocessed data through the lstm
        lstm_h, lstm_c = self.clstm(x=pre_act)

        # Pass the lstm output through the postprocessing layer
        post_act = self.post_layer(lstm_h)

        # Dynamic output
        dyn_out = post_act[:, :self.params.pk_dyn_out_size]

        # Lateral output
        lat_out = th.tanh(post_act[:, -self.params.pk_lat_size:])

        # Update the output and hidden state tensors of the PKs
        self.tensors.pk_dyn_out = dyn_out
        self.tensors.pk_lat_out = lat_out
        self.tensors.pk_lstm_h = lstm_h
        self.tensors.pk_lstm_c = lstm_c
        
    def reset(self, num_of_pks):
        self.tensors.reset(num_of_pks=num_of_pks)
        self.clstm.reset_states()

    def detach(self):
        self.tensors.detach()
