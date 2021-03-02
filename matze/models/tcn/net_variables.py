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
        self.batch_size = cfg.BATCH_SIZE
        self.input_size = cfg.INPUT_SIZE
        self.field_size = cfg.FIELD_SIZE

        # First convolution layer (16x16 -> 16x16)
        self.tconv_num_channels = cfg.TCONV_NUM_CHANNELS
        self.tconv_kernel_size = cfg.TCONV_KERNEL_SIZE


class NetworkTensors:
    """
    This class holds the tensors of the Network.
    """

    def __init__(self, _params):
        self.params = _params

        # Create the tensors by calling the reset method
        self.reset(self.params.batch_size)

    def reset(self, batch_size):

        #
        # Network tensors

        # Inputs
        self.inputs = th.zeros(size=(batch_size,
                                     self.params.seq_len,
                                     self.params.field_size,
                                     self.params.input_size),
                               device=self.params.device)

        # Outputs
        self.outputs = th.zeros(size=(batch_size,
                                      self.params.seq_len,
                                      self.params.field_size,
                                      self.params.input_size),
                                device=self.params.device)
