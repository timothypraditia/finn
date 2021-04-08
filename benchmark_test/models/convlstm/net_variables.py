import torch as th
import configuration as cfg


class NetworkParameters:
    """
    This class holds the parameters of the Network.
    """

    def __init__(self, device):

        #
        # System parameters
        self.device = device

        #
        # Network parameters
        self.seq_len = cfg.SEQ_LEN
        self.batches = cfg.BATCH_SIZE
        self.input_size = cfg.INPUT_SIZE
        self.field_size = cfg.FIELD_SIZE

        # First convolution layer (16x16 -> 16x16)
        self.convlstm1_input_size = cfg.CONVLSTM1_INPUT_SIZE
        self.convlstm1_input_channels = cfg.CONVLSTM1_INPUT_CHANNELS
        self.convlstm1_hidden_channels = cfg.CONVLSTM1_HIDDEN_CHANNELS
        self.convlstm1_kernel_size = cfg.CONVLSTM1_KERNEL_SIZE
        self.convlstm1_bias = cfg.CONVLSTM1_USE_BIAS

        # Second convolution layer (16x16 -> 16x16)
        self.convlstm2_input_size = cfg.CONVLSTM2_INPUT_SIZE
        self.convlstm2_input_channels = cfg.CONVLSTM2_INPUT_CHANNELS
        self.convlstm2_hidden_channels = cfg.CONVLSTM2_HIDDEN_CHANNELS
        self.convlstm2_kernel_size = cfg.CONVLSTM2_KERNEL_SIZE
        self.convlstm2_bias = cfg.CONVLSTM2_USE_BIAS


class NetworkTensors:
    """
    This class holds the tensors of the Network.
    """

    def __init__(self, _params):
        self.params = _params

        # Create the tensors by calling the reset method
        self.reset(self.params.batches)

    def reset(self, batches):

        #
        # Network tensors

        # Inputs
        self.inputs = th.zeros(size=(batches,
                                     self.params.input_size,
                                     self.params.field_size),
                               device=self.params.device)

        # ConvLSTM states
        self.convlstm1_c = th.zeros(size=(batches,
                                          self.params.convlstm1_hidden_channels,
                                          self.params.convlstm1_input_size),
                                   device=self.params.device)
        self.convlstm1_h = th.zeros(size=(batches,
                                          self.params.convlstm1_hidden_channels,
                                          self.params.convlstm1_input_size),
                                   device=self.params.device)

        self.convlstm2_c = th.zeros(size=(batches,
                                          self.params.convlstm2_hidden_channels,
                                          self.params.convlstm2_input_size),
                                    device=self.params.device)

        self.convlstm2_h = th.zeros(size=(batches,
                                          self.params.convlstm2_hidden_channels,
                                          self.params.convlstm1_input_size),
                                    device=self.params.device)

        # Outputs
        self.outputs = th.zeros(size=(batches,
                                      self.params.input_size,
                                      self.params.field_size),
                                device=self.params.device)
