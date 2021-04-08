# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Set_Module class that constructs separate objects for
different core samples, based on the parameters read from the Excel input file
"""

import torch
import numpy as np
import pandas as pd


class Set_Module:
    
    def __init__(self, filename, params):
        """
        Constructor
        
        Inputs:
            filename    : the corresponding filename for the core sample
            params      : the configuration object containing the model settings

        """

        # Load parameters from the Excel file
        in_params = pd.read_excel(filename, sheet_name=1, index_col=0, header=None)
        
        # Determine the device on which the training takes place
        self.device = params.device
        
        # Soil Parameters
        self.D = in_params[1]['D']
        self.por = in_params[1]['por']
        self.rho_s = in_params[1]['rho_s']
        
        
        # Simulation Domain
        self.X = in_params[1]['X']
        self.Nx = int(in_params[1]['Nx'])
        self.dx = self.X / (self.Nx+1)
        self.T = in_params[1]['T']
        self.r = in_params[1]['sample_radius']
        self.A = np.pi * self.r**2
        self.Q = in_params[1]['Q']
        self.solubility = in_params[1]['solubility']
        self.cauchy_val = self.por * self.A / self.Q * self.dx
        
        
        # Inputs for Flux Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_flux = [params.flux_layers, params.flux_layers]
        self.num_nodes_flux = [params.flux_nodes, params.flux_nodes]
        ## Set numerical stencil to be learnable or not
        self.learn_stencil = [False, False]
        ## Effective diffusion coefficient for each variable
        self.D_eff = [self.D / (self.dx**2), self.D * self.por / (self.rho_s/1000) / (self.dx**2)]
        ## Set diffusion coefficient to be learnable or not
        self.learn_coeff = [False, False]
        ## Set if diffusion coefficient to be approximated as a function
        self.coeff_func = [True, False]
        ## Normalizer for functions that are approximated with a NN
        self.p_exp_flux = [torch.tensor([0.0]), torch.tensor([0.0])]
        ## Set the variable index to be used when calculating the fluxes
        self.flux_calc_idx = [torch.tensor([0]), torch.tensor([0])]
        ## Set the variable indices necessary to calculate the diffusion
        ## coefficient function
        self.flux_couple_idx = [torch.tensor([0]), torch.tensor([0])]
        ## Set boundary condition types
        self.dirichlet_bool = [[True, np.bool(in_params[1]['Dirichlet']), False, False],
                               [True, np.bool(in_params[1]['Dirichlet']), False, False]]
        self.neumann_bool = [[False, False, True, True], [False, False, True, True]]
        self.cauchy_bool = [[False, np.bool(in_params[1]['Cauchy']), False, False],
                            [False, np.bool(in_params[1]['Cauchy']), False, False]]
        ## Set the Dirichlet and Neumann boundary values if necessary,
        ## otherwise set = 0
        self.dirichlet_val = [torch.tensor([self.solubility, 0.0, 0.0, 0.0]),
                              torch.tensor([self.solubility, 0.0, 0.0, 0.0])]
        self.neumann_val = [torch.tensor([0.0, 0.0, 0.0, 0.0]),
                            torch.tensor([0.0, 0.0, 0.0, 0.0])]
        ## Set multiplier for the Cauchy boundary condition if necessary
        ## (will be multiplied with D_eff in the flux kernels), otherwise set = 0
        self.cauchy_mult = [self.cauchy_val, self.cauchy_val]
        
        
        # Inputs for State Kernels
        ## Set number of hidden layers and hidden nodes
        self.num_layers_state = [params.state_layers, params.state_layers]
        self.num_nodes_state = [params.state_nodes, params.state_nodes]
        ## Set if there is any reaction to be approximated as a function
        self.react_func = [False, False]
        ## Normalizer for the reaction functions that are approximated with a NN
        self.p_exp_state = [torch.tensor([0.0]), torch.tensor([0.0])]
        ## Set the variable indices necessary to calculate the reaction function
        self.state_couple_idx = [torch.tensor([0]), torch.tensor([1])]