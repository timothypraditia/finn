# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains all inputs necessary to configure and set up the model and
simulation
"""

import torch

# MODEL NAME & SETTING
model_name = "syn_freundlich_01"
save_model = True
continue_training = False # Set to True to continue training with a saved model
device_name = "cpu" # Choose between "cpu" or "cuda"


# NETWORK HYPER-PARAMETERS
flux_layers = 3         # number of hidden layers for the NN in the flux kernels
state_layers = 3        # number of hidden layers for the NN in the state kernels
flux_nodes = 15         # number of hidden nodes per layer for the NN in the flux kernels
state_nodes = 15        # number of hidden nodes per layer for the NN in the flux kernels
learning_rate = 0.1     # learning rate for the optimizer
error_mult = 1          # multiplier for the squared error in the loss function calculation
breakthrough_mult = 1   # multiplier for the breakthrough curve error in the loss function calculation
profile_mult = 1        # multiplier for the concentration profile error in the loss function calculation
phys_mult = 100         # multiplier for the physical regularization in the loss function calculation
epochs = 100            # maximum epoch for training
lbfgs_optim = True      # Use L-BFGS as optimizer, else use ADAM
train_breakthrough = False # Train using only breakthrough curve data
linear = False          # Training data generated with the linear isotherm
freundlich = True       # Training data generated with the freundlich isotherm
langmuir = False        # Training data generated with the langmuir isotherm


# SIMULATION-RELATED INPUTS
num_vars = 2

# Soil Parameters
D = 0.0005              # effective diffusion coefficient [m^2/day]
por = 0.29              # porosity [-]
rho_s = 2880            # bulk density [kg/m^3]
Kf = 1.016/rho_s        # freundlich's K [(m^3/kg)^nf]
nf = 0.874              # freundlich exponent [-]
smax = 1/1700           # sorption capacity [m^3/kg]
Kl = 1                  # half-concentration [kg/m^3]
Kd = 0.429/1000         # organic carbon partitioning [m^3/kg]
solubility = 1.0        # top boundary value [kg/m^3]

# Simulation Domain
X = 1.0                 # length of sample [m]
dx = 0.04               # length of discrete control volume [m]
T = 10000               # simulation time [days]
dt = 5                  # time step [days]

# Inputs for Flux Kernels
## Set number of hidden layers and hidden nodes
num_layers_flux = [flux_layers, flux_layers]
num_nodes_flux = [flux_nodes, flux_nodes]
## Set numerical stencil to be learnable or not
learn_stencil = [False, False]
## Effective diffusion coefficient for each variable
# D_eff = [D / (dx**2), D * por / (rho_s/1000) / (dx**2)]
D_eff = [D / (dx**2), 0.25]
## Set diffusion coefficient to be learnable or not
learn_coeff = [False, True]
## Set if diffusion coefficient to be approximated as a function
coeff_func = [True, False]
## Normalizer for functions that are approximated with a NN
p_exp_flux = [torch.tensor([0.0]), torch.tensor([0.0])]
## Set the variable index to be used when calculating the fluxes
flux_calc_idx = [torch.tensor([0]), torch.tensor([0])]
## Set the variable indices necessary to calculate the diffusion
## coefficient function
flux_couple_idx = [torch.tensor([0]), torch.tensor([0])]
## Set boundary condition types
dirichlet_bool = [[True, False, False, False],
                  [True, False, False, False]]
neumann_bool = [[False, False, True, True],
                [False, False, True, True]]
cauchy_bool = [[False, True, False, False],
               [False, True, False, False]]
## Set the Dirichlet and Neumann boundary values if necessary,
## otherwise set = 0
dirichlet_val = [torch.tensor([solubility, 0.0, 0.0, 0.0]),
                 torch.tensor([solubility, 0.0, 0.0, 0.0])]
neumann_val = [torch.tensor([0.0, 0.0, 0.0, 0.0]),
               torch.tensor([0.0, 0.0, 0.0, 0.0])]
## Set multiplier for the Cauchy boundary condition if necessary
## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
cauchy_mult = [dx, dx]

# Inputs for State Kernels
## Set number of hidden layers and hidden nodes
num_layers_state = [state_layers, state_layers]
num_nodes_state = [state_nodes, state_nodes]
## Set if there is any reaction to be approximated as a function
react_func = [False, False]
## Normalizer for the reaction functions that are approximated with a NN
p_exp_state = [torch.tensor([0.0]), torch.tensor([0.0])]
## Set the variable indices necessary to calculate the reaction function
state_couple_idx = [torch.tensor([0]), torch.tensor([1])]