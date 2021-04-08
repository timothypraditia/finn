# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Initialize class that reads the configuration file
and construct an object with the corresponding configuration parameters
"""

import torch
import os
import shutil
import syn00_config as params

class Initialize:
    
    def __init__(self):
        """
        Constructor
        
        """
        
        # SET WORKING PATH
        self.main_path = os.getcwd()
        
        # MODEL NAME & SETTING
        self.model_name = params.model_name
        
        self.model_path = self.main_path + '\\' + self.model_name
        self.check_dir(self.model_path)
        
        self.log_path = self.main_path + '\\runs\\' + self.model_name
        # Remove old log files to prevent unclear visualization in tensorboard
        self.check_dir(self.log_path, remove=True)
        
        self.save_model = params.save_model
        self.continue_training = params.continue_training
        self.device_name = params.device_name
        self.device = self.determine_device()

        # NETWORK HYPER-PARAMETERS
        self.flux_layers = params.flux_layers
        self.state_layers = params.state_layers
        self.flux_nodes = params.flux_nodes
        self.state_nodes = params.state_nodes
        self.learning_rate = params.learning_rate
        self.error_mult = params.error_mult
        self.breakthrough_mult = params.breakthrough_mult
        self.profile_mult = params.profile_mult
        self.phys_mult = params.phys_mult
        self.epochs = params.epochs
        self.lbfgs_optim = params.lbfgs_optim
        self.train_breakthrough = params.train_breakthrough
        self.linear = params.linear
        self.freundlich = params.freundlich
        self.langmuir = params.langmuir
        
        # SIMULATION-RELATED INPUTS
        self.num_vars = params.num_vars
        
        # Soil Parameters
        self.D = params.D
        self.por = params.por
        self.rho_s = params.rho_s
        self.Kf = params.Kf
        self.nf = params.nf
        self.smax = params.smax
        self.Kl = params.Kl
        self.Kd = params.Kd
        self.solubility = params.solubility
        
        # Simulation Domain
        self.X = params.X
        self.dx = params.dx
        self.Nx = int(self.X / self.dx + 1)
        self.T = params.T
        self.dt = params.dt
        self.Nt = int(self.T / self.dt + 1)
        self.cauchy_val = self.dx
        
        # Inputs for Flux Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_flux = params.num_layers_flux
        self.num_nodes_flux = params.num_nodes_flux
        ## Set numerical stencil to be learnable or not
        self.learn_stencil = params.learn_stencil
        ## Effective diffusion coefficient for each variable
        self.D_eff = params.D_eff
        ## Set diffusion coefficient to be learnable or not
        self.learn_coeff = params.learn_coeff
        ## Set if diffusion coefficient to be approximated as a function
        self.coeff_func = params.coeff_func
        ## Normalizer for functions that are approximated with a NN
        self.p_exp_flux = params.p_exp_flux
        ## Set the variable index to be used when calculating the fluxes
        self.flux_calc_idx = params.flux_calc_idx
        ## Set the variable indices necessary to calculate the diffusion
        ## coefficient function
        self.flux_couple_idx = params.flux_couple_idx
        ## Set boundary condition types
        self.dirichlet_bool = params.dirichlet_bool
        self.neumann_bool = params.neumann_bool
        self.cauchy_bool = params.cauchy_bool
        ## Set the Dirichlet and Neumann boundary values if necessary,
        ## otherwise set = 0
        self.dirichlet_val = params.dirichlet_val
        self.neumann_val = params.neumann_val
        ## Set multiplier for the Cauchy boundary condition if necessary
        ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
        self.cauchy_mult = params.cauchy_mult
        
        # Inputs for State Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_state = params.num_layers_state
        self.num_nodes_state = params.num_nodes_state
        ## Set if there is any reaction to be approximated as a function
        self.react_func = params.react_func
        ## Normalizer for the reaction functions that are approximated with a NN
        self.p_exp_state = params.p_exp_state
        ## Set the variable indices necessary to calculate the reaction function
        self.state_couple_idx = params.state_couple_idx
    
    
    def determine_device(self):
        """
        This function evaluates whether a GPU is accessible at the system and
        returns it as device to calculate on, otherwise it returns the CPU.
        :return: The device where tensor calculations shall be made on
        """
        
        self.device = torch.device(self.device_name)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        print("Using device:", self.device)
        print()
        
        # Additional Info when using cuda
        if self.device.type == "cuda" and torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print("\tAllocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
            print("\tCached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
            print()
        
        return self.device
    
    
    def check_dir(self, path_dir, remove=False):
        """
        This function evaluates whether a directory for the corresponding model
        exists, otherwise create a new directory
        
        Inputs:
            path_dir    : the path to the corresponding directory
            remove      : a Boolean value to determine whether to delete pre-
                            existing directory
        """
        
        # For tensorboard log files, clean the directory to avoid error in
        # visualization
        if remove:
            if os.path.exists(path_dir):
                shutil.rmtree(path_dir)
            
        # If the path does not exist, create a new path
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)