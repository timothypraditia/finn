import numpy as np
import torch as th
import time
import glob
import os
import matplotlib.pyplot as plt
import net_variables
import net_model
import configuration as cfg
from utils import utils

th.set_num_threads(1)

# Hide the GPU(s) in case the user specified to use the CPU in the config file
if cfg.DEVICE == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_testing(model_name, sequence_length, data_noise, data_file,
                visualize_results):

    # Set a globally reachable boolean in the config file for training
    cfg.TRAINING = False
    cfg.MODEL_NAME = "convlstm" + str(model_name) + "_noise=" + str(data_noise)
    cfg.SEQ_LEN = sequence_length
    cfg.DATA_NOISE = data_noise

    # setting device on GPU if available, else CPU
    device = utils.determine_device()

    # Set up the parameter and tensor classes
    params = net_variables.NetworkParameters(
        device=device
    )
    tensors = net_variables.NetworkTensors(_params=params)

    # Initialize and set up the kernel network
    net = net_model.Model(
        _params=params,
        _tensors=tensors
    )

    # Restore the network by loading the weights saved in the .pt file
    print('Restoring model (that is the network\'s weights) from file...')
    net.load_state_dict(th.load("models/convlstm/saved_models/"
                                + cfg.MODEL_NAME + "/" + cfg.MODEL_NAME
                                + ".pt",
                                map_location=device))
    net.eval()

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    """
    TESTING
    """

    #
    # Get the training and validation file names
    data_filename = "data.npy" if data_file == "seen" else "data_test.npy"
    data = np.array(np.load("data/numpy/" + data_filename),
                    dtype=np.float32)[:cfg.SEQ_LEN + 1]
    data = np.swapaxes(data, axis1=1, axis2=2)
    data = np.expand_dims(data, axis=1)

    time_start = time.time()

    # Separate the training data into network inputs and labels
    net_input = data[:-1]
    net_label = data[1:]

    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(params.seq_len,
                                 params.batches,
                                 params.input_size,
                                 params.field_size))

    # Reset the network to clear the previous sequence
    net.reset(batches=params.batches)

    # Iterate over the whole sequence of the training example and perform a
    # forward pass
    for t in range(params.seq_len):

        # Prepare the network input for this sequence step
        if t > cfg.TEACHER_FORCING_STEPS:
            # Closed loop - receiving the output of the last time step as input
            net_input_step = net_outputs[t-1].detach().numpy()
        else:
            net_input_step = net_input[t]

        # Forward the input through the network
        net.forward(net_in=net_input_step)

        # Store the output of the network for this sequence step
        net_outputs[t] = tensors.output

    forward_pass_duration = time.time() - time_start
    print("Forward pass took:", forward_pass_duration, "seconds.")

    # Transform the PyTorch outputs tensor into a numpy array
    net_outputs = net_outputs.detach().numpy()

    # Bring data back to original shape
    net_outputs = np.swapaxes(net_outputs[:, 0], axis1=1, axis2=2)
    net_outputs = np.swapaxes(net_outputs, axis1=0, axis2=1)
    net_label = np.swapaxes(net_label[:, 0], axis1=1, axis2=2)
    net_label = np.swapaxes(net_label, axis1=0, axis2=1)

    return net_outputs, net_label
