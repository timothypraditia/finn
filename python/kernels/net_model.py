# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Net_Model class, which sets up modules of flux and 
state kernels and combines them for coupled calculation of du/dt
"""

import torch
import flux_kernels
import state_kernels

class Net_Model(torch.nn.Module):
    
    def __init__(self, u0, cfg):
        
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [num_features, Nx, Ny]
            cfg     : configuration object of the model setup, containing boundary
                        condition types, values, learnable parameter settings, etc.
        """
        
        super(Net_Model, self).__init__()
        
        # Initialize the flux and state kernels as an empty module list
        self.flux_modules = torch.nn.ModuleList()
        self.state_modules = torch.nn.ModuleList()
        
        # Extract information of number of variables to be calculated
        self.num_vars = u0.size(0)
        
        # Extract configuration information
        self.cfg = cfg
        
        # Create flux and state kernel for each variable to be calculated
        for var_idx in range(self.num_vars):
            self.flux_modules.append(flux_kernels.Flux_Kernels(u0[var_idx], self.cfg, var_idx))
            self.state_modules.append(state_kernels.State_Kernels(self.cfg, var_idx))

        
    
    def forward(self, t, u):
        
        """
        The forward function calculates du/dt to be put into the ODE solver
        
        Inputs:
            t   : time (scalar value, taken from the ODE solver)
            u   : the unknown variables to be calculated taken from the previous
                    time step, dim: [num_features, Nx, Ny]
            
        Output:
            du  : the time derivative of u (du/dt), dim: [num_features, Nx, Ny]
            
        """
        
        # Initialize the flux and state kernel outputs in empty lists
        flux = []
        state = []
        
        # Use flux and state kernels to calculate du/dt for all unknown variables
        for var_idx in range(self.num_vars):
            flux.append(self.flux_modules[var_idx](u[self.cfg.flux_calc_idx[var_idx]],
                                                   u[self.cfg.flux_couple_idx[var_idx]], t))
            state.append(self.state_modules[var_idx](u[self.cfg.state_couple_idx[var_idx]],
                                                     flux[var_idx]))
        
        # Convert the state list into a torch.tensor type
        du = torch.stack(state)
            
        return du
    
    
