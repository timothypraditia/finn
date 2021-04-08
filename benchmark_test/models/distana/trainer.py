import numpy as np
import torch as th
import torch.nn as nn
import glob
import os
import time
import matplotlib.pyplot as plt
from threading import Thread
import kernel_variables
from distana import DISTANA
import configuration as cfg
from utils import utils

th.set_num_threads(1)

# Hide the GPU(s) in case the user specified to use the CPU in the config file
if cfg.DEVICE == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_training(model_name, print_progress, sequence_length, data_noise):

    # Set a globally reachable boolean in the config file for training
    cfg.TRAINING = True
    cfg.MODEL_NAME = "distana" + str(model_name) + "_noise=" + str(data_noise)
    cfg.SEQ_LEN = sequence_length
    cfg.DATA_NOISE = data_noise

    # Set a random seed for varying weight initializations
    th.seed()

    # Read the configuration file to be able to save it later if desired
    if cfg.SAVE_MODEL:
        with open("models/distana/configuration.py", "r") as f:
            cfg_file = f.read()

    # Print some information to console
    print("Architecture name:", cfg.ARCHITECTURE_NAME)
    print("Model name:", cfg.MODEL_NAME)

    # Set device on GPU if specified in the configuration file, else CPU
    device = utils.determine_device()

    # Compute batch size for the PKs (every PK is processed in a separate batch
    # to parallelize computation)
    num_of_pks = cfg.FIELD_SIZE

    # Set up the parameter and tensor classes
    params = kernel_variables.KernelParameters(
        num_of_pks=num_of_pks,
        device=device
    )
    tensors = kernel_variables.KernelTensors(params=params)

    # Initialize and set up the DISTANA model
    model = DISTANA(
        params=params,
        tensors=tensors
    )

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    #
    # Get the training file names
    data = np.array(np.load("data/numpy/data.npy"),
                    dtype=np.float32)[:cfg.SEQ_LEN + 1]
    data = np.swapaxes(data, axis1=1, axis2=2)
    data = np.expand_dims(data, axis=1)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if cfg.CONTINUE_TRAINING:
        print("Restoring model (that is the network\"s weights) from file...")
        net.load_state_dict(th.load("saved_models/" + cfg.MODEL_NAME + "/"
                                    + cfg.MODEL_NAME + ".pt",
                                    map_location=device))
        net.train()

    """
    TRAINING
    """

    a = time.time()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors = []
    best_train = np.infty

    #
    # Start the training and iterate over all epochs
    for epoch in range(cfg.EPOCHS):

        epoch_start_time = time.time()
        sequence_errors = []

        # Iterate over all training iterations and evaluate the network
        for train_iter in range(cfg.TRAINING_ITERS):

            # Generate the training data batch for this iteration
            net_input, net_label = data[:-1], data[1:]

            # Set up an array of zeros to store the network outputs
            net_outputs = th.zeros(size=(cfg.SEQ_LEN,
                                         params.batch_size,
                                         params.pk_dyn_out_size,
                                         params.field_size))

            # Set the gradients back to zero
            optimizer.zero_grad()

            # Reset the network to clear the previous sequence
            model.reset(num_of_pks=num_of_pks)

            # Iterate over the whole sequence of the training example and
            # perform a forward pass
            for t in range(cfg.SEQ_LEN):

                # Teacher forcing - Set the input for this iteration
                net_in_step = net_input[t]

                # Forward the input through the network
                model.forward(dyn_in=net_in_step)

                # Store the output of the network for this sequence step
                net_outputs[t] = tensors.pk_dyn_out

            mse = None

            # Get the mean squared error from the evaluation list
            mse = criterion(net_outputs, th.from_numpy(net_label))
            
            # Backpropagate the error and perform a weight update
            mse.backward()
            optimizer.step()

            # Append the current error to the sequence_errors list
            sequence_errors.append(mse.item())

        epoch_errors.append(np.mean(sequence_errors))

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors[-1]

            if cfg.SAVE_MODEL:
                # Start a separate thread to save the model
                thread = Thread(target=utils.save_model_to_file(
                    path="models/distana/saved_models/" + cfg.MODEL_NAME
                         + "/",
                    cfg_file=cfg_file,
                    current_epoch=epoch,
                    epochs=cfg.EPOCHS,
                    epoch_errors=epoch_errors,
                    net=model,
                    model_name=cfg.MODEL_NAME))
                thread.start()

        #
        # Print progress to the console
        if print_progress:
            print("Epoch "
                  + str(epoch + 1).zfill(int(np.log10(cfg.EPOCHS)) + 1)
                  + "/" + str(cfg.EPOCHS) + " took "
                  + str(np.round(time.time() - epoch_start_time,
                        2)).ljust(5, "0")
                  + " seconds.\t\tAverage epoch training error: "
                  + train_sign
                  + str(np.round(epoch_errors[-1], 10)).ljust(12, " "))

    b = time.time()
    print("\nTraining took " + str(np.round(b - a, 2)) + " seconds.\n\n")

    return epoch_errors
