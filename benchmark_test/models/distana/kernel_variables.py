import torch as th
import configuration as cfg


class KernelParameters:
    """
    This class holds the parameters of the Kernel Network.
    """

    def __init__(self, num_of_pks, device):

        #
        # System parameters
        self.device = device

        #
        # Network parameters
        self.seq_len = cfg.SEQ_LEN
        self.batch_size = cfg.BATCH_SIZE
        self.input_size = cfg.INPUT_SIZE
        self.field_size = cfg.FIELD_SIZE

        #
        # PK specific parameters
        self.num_of_pks = num_of_pks
        
        # Convolution parameters
        self.pk_conv_ksize = cfg.PK_CONV_KSIZE
        self.pk_conv_stride = cfg.PK_CONV_STRIDE
        self.pk_conv_padding = cfg.PK_CONV_PADDING

        # Input sizes (dimensions)
        self.pk_dyn_in_size = cfg.PK_DYN_IN_SIZE
        
        # Layer sizes (number of neurons per layer)
        self.pk_pre_layer_size = cfg.PK_PRE_LAYER_SIZE
        self.pk_num_lstm_cells = cfg.PK_NUM_LSTM_CELLS

        # Output sizes (dimensions)
        self.pk_dyn_out_size = cfg.PK_DYN_OUT_SIZE

        # Lateral vector size
        self.pk_lat_size = cfg.PK_LAT_SIZE


class KernelTensors:
    """
    This class holds the tensors of the Kernel Network.
    """

    def __init__(self, params):
        self.params = params

        # Initialize the tensors of the PK

        # Inputs
        self.pk_dyn_in = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_dyn_in_size,
                  self.params.field_size),
            device=self.params.device
        )
        self.pk_lat_in = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_lat_size,
                  self.params.field_size),
            device=self.params.device
        )

        # LSTM states
        self.pk_lstm_c = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_num_lstm_cells,
                  self.params.field_size),
            device=self.params.device,
        )
        self.pk_lstm_h = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_num_lstm_cells,
                  self.params.field_size),
            device=self.params.device,
        )

        # Outputs
        self.pk_dyn_out = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_dyn_out_size,
                  self.params.field_size),
            device=self.params.device
        )
        self.pk_lat_out = th.zeros(
            size=(self.params.batch_size,
                  self.params.pk_lat_size,
                  self.params.field_size),
            device=self.params.device
        )

    def reset(self, num_of_pks):
        # Inputs
        self.pk_dyn_in = th.zeros_like(self.pk_dyn_in)
        self.pk_lat_in = th.zeros_like(self.pk_lat_in)

        # LSTM states
        self.pk_lstm_c = th.zeros_like(self.pk_lstm_c)
        self.pk_lstm_h = th.zeros_like(self.pk_lstm_h)

        # Outputs
        self.pk_dyn_out = th.zeros_like(self.pk_dyn_out)
        self.pk_lat_out = th.zeros_like(self.pk_lat_out)

    def detach(self):
        # Inputs
        self.pk_dyn_in = self.pk_dyn_in.detach()
        self.pk_lat_in = self.pk_lat_in.detach()

        # LSTM states
        self.pk_lstm_c = self.pk_lstm_c.detach()
        self.pk_lstm_h = self.pk_lstm_h.detach()

        # Outputs
        self.pk_dyn_out = self.pk_dyn_out.detach()
        self.pk_lat_out = self.pk_lat_out.detach()
