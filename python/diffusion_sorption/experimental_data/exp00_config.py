# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains all inputs necessary to configure and set up the model and
simulation
"""

# MODEL NAME & SETTING
model_name = "exp_01"
save_model = True
continue_training = False # Set to True to continue training with a saved model
device_name = "cpu" # Choose between "cpu" or "cuda"


# NETWORK HYPER-PARAMETERS
flux_layers = 4         # number of hidden layers for the NN in the flux kernels
state_layers = 4        # number of hidden layers for the NN in the state kernels
flux_nodes = 15         # number of hidden nodes per layer for the NN in the flux kernels
state_nodes = 15        # number of hidden nodes per layer for the NN in the state kernels
learning_rate = 0.1     # learning rate for the optimizer
error_mult = 1e5        # multiplier for the squared error in the loss function calculation
phys_mult = 100         # multiplier for the physical regularization in the loss function calculation
epochs = 100            # maximum epoch for training
lbfgs_optim = True      # Use L-BFGS as optimizer, else use ADAM


# SIMULATION-RELATED INPUTS
num_vars = 2            # number of unknown variables to be calculated