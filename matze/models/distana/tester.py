import numpy as np
import torch as th
import os
import time
import kernel_variables
from distana import DISTANA
import configuration as cfg
import sys
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
    cfg.MODEL_NAME = "distana" + str(model_name) + "_noise=" + str(data_noise)
    cfg.SEQ_LEN = sequence_length
    cfg.DATA_NOISE = data_noise

    # setting device on GPU if available, else CPU
    device = utils.determine_device()

    # Compute batch size for PKs and TKs
    num_of_pks = cfg.FIELD_SIZE

    # Set up the parameter and tensor classes
    params = kernel_variables.KernelParameters(
        num_of_pks=num_of_pks,
        device=device
    )
    # tensors1 = kernel_variables.KernelTensors(_params=params)
    tensors = kernel_variables.KernelTensors(params=params)

    # Initialize and set up the kernel network
    model = DISTANA(
        params=params,
        tensors=tensors
    )

    # Restore the network by loading the weights saved in the .pt file
    print('Restoring model (that is the network\'s weights) from file...')
    model.load_state_dict(th.load("models/distana/saved_models/"
                                  + cfg.MODEL_NAME + "/" + cfg.MODEL_NAME
                                  + ".pt",
                                  map_location=device))
    model.eval()

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    """
    TESTING
    """

    #
    # Get the test file names
    data_filename = "data.npy" if data_file == "seen" else "data_test.npy"
    data = np.array(np.load("data/numpy/" + data_filename),
                    dtype=np.float32)[:cfg.SEQ_LEN + 1]
    data = np.swapaxes(data, axis1=1, axis2=2)
    data = np.expand_dims(data, axis=1)

    time_start = time.time()

    #
    # Evaluate the network for the given test data

    # Separate the data into network input and labels
    net_input, net_label = data[:-1], data[1:]

    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(sequence_length,
                                 params.batch_size,
                                 params.pk_dyn_out_size,
                                 params.field_size))

    # Iterate over the whole sequence of the training example and perform a
    # forward pass
    for t in range(sequence_length):

        # Prepare the network input for this sequence step
        if t > cfg.TEACHER_FORCING_STEPS:
            #
            # Closed loop - receiving the output of the last time step as
            # input
            net_in_step = net_outputs[t - 1].detach().numpy()
        else:
            #
            # Teacher forcing - Set the dynamic input for this iteration
            net_in_step = net_input[t]

        # Forward the input through the network
        model.forward(dyn_in=net_in_step)

        # Store the output of the network for this sequence step
        net_outputs[t] = tensors.pk_dyn_out

    # Calculate forward pass duration and dump it to console
    forward_pass_duration = time.time() - time_start
    print("\tForward pass took:", forward_pass_duration, "seconds.")

    # Convert the net_outputs tensor into a numpy array
    net_outputs = net_outputs.detach().numpy()

    # Bring data back to original shape
    net_outputs = np.swapaxes(net_outputs[:, 0], axis1=1, axis2=2)
    net_outputs = np.swapaxes(net_outputs, axis1=0, axis2=1)
    net_label = np.swapaxes(net_label[:, 0], axis1=1, axis2=2)
    net_label = np.swapaxes(net_label, axis1=0, axis2=1)

    return net_outputs, net_label
