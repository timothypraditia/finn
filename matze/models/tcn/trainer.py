import numpy as np
import torch as th
import torch.nn as nn
import time
import glob
import os
import matplotlib.pyplot as plt
from threading import Thread
import net_variables
import net_model
import configuration as cfg
from utils import utils

th.set_num_threads(1)

# Hide the GPU(s) in case the user specified to use the CPU in the config file
if cfg.DEVICE == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_training(model_name, print_progress, sequence_length, data_noise):

    # Set a globally reachable boolean in the config file for training
    cfg.TRAINING = True
    cfg.MODEL_NAME = "tcn" + str(model_name) + "_noise=" + str(data_noise)
    cfg.SEQ_LEN = sequence_length
    cfg.DATA_NOISE = data_noise

    # Set a random seed for varying weight initializations
    th.seed()

    # Load the configuration file to be able to save it later if desired
    if cfg.SAVE_MODEL:
        with open("models/tcn/configuration.py", "r") as f:
            cfg_file = f.read()

    print("Architecture name:", cfg.ARCHITECTURE_NAME)
    print("Model name:", cfg.MODEL_NAME)

    time_start = time.time()

    # setting device on GPU if available, else CPU
    device = utils.determine_device()

    # Set up the parameter and tensor classes
    params = net_variables.NetworkParameters(
        device=device
    )
    tensors = net_variables.NetworkTensors(_params=params)

    # Initialize and set up the network
    net = net_model.Model(
        _params=params,
        _tensors=tensors
    )

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors = []
    best_train = np.infty

    #
    # Get the training and validation file names
    data = np.array(np.load("data/numpy/data.npy"),
                    dtype=np.float32)[:cfg.SEQ_LEN + 1]
    data = np.swapaxes(data, axis1=1, axis2=2)
    data = np.swapaxes(data, axis1=0, axis2=1)
    data = np.expand_dims(data, axis=0)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if cfg.CONTINUE_TRAINING:
        print("Restoring model (that is the network\"s weights) from file...")
        net.load_state_dict(th.load("models/tcn/saved_models/" + cfg.MODEL_NAME
                                    + "/" + cfg.MODEL_NAME + ".pt"))
        net.train()

    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(cfg.EPOCHS):

        epoch_start_time = time.time()

        sequence_errors = []

        # Iterate over all training iterations and evaluate the network
        for train_iter in range(cfg.TRAINING_ITERS):

            # Separate the data into network inputs and labels
            net_input = data[:, :, :-1]
            net_label = data[:, :, 1:]
            
            optimizer.zero_grad()

            # Reset the network to clear the previous sequence
            net.reset(batch_size=params.batch_size)

            # Forward the input through the network
            net.forward(net_in=net_input)
            # Store the output of the network for this sequence step
            net_outputs = tensors.output

            # Comput the error
            mse = criterion(
                net_outputs[:, :, :],
                th.from_numpy(net_label[:, :, :]).to(device=params.device)
            )

            # Backpropagate the error and perform a weight update
            mse.backward()
            optimizer.step()

            sequence_errors.append(mse.item())

        epoch_errors.append(np.mean(sequence_errors))

        train_sign = "(-)"
        if epoch_errors[-1] < best_train:
            best_train = epoch_errors[-1]
            train_sign = "(+)"

        #
        # Print progress to the console
        print("Epoch "
              + str(epoch + 1).zfill(int(np.log10(cfg.EPOCHS)) + 1)
              + "/" + str(cfg.EPOCHS) + " took "
              + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, "0")
              + " seconds.\t\tAverage epoch training error: "
              + train_sign
              + str(np.round(epoch_errors[-1], 20)).ljust(12, " "))

        # Save the model to file (if desired)
        if cfg.SAVE_MODEL and train_sign == "(+)":
            # Start a separate thread to save the model
            thread = Thread(target=utils.save_model_to_file(
                path="models/tcn/saved_models/" + cfg.MODEL_NAME + "/",
                cfg_file=cfg_file,
                current_epoch=epoch,
                epochs=cfg.EPOCHS,
                epoch_errors=epoch_errors,
                net=net,
                model_name=cfg.MODEL_NAME))
            thread.start()

    b = time.time()
    print("\nTraining took " + str(np.round(b - a, 2)) + " seconds.\n\n")

    return epoch_errors
