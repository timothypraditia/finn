# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Flux_Kernels class, which takes input of ui and its
neighbors, and returns the integrated flux approximation
"""

import torch


class Flux_Kernels(torch.nn.Module):
    
    def __init__(self, u0, cfg, var_idx):
        
        """
        Constructor
        Inputs:
            u0      : initial condition, dim: [Nx, Ny]
            cfg     : configuration object of the model setup, containing boundary
                        condition types, values, learnable parameter settings, etc.
            var_idx : index of the calculated variable (could be > 1 for coupled
                        systems)
        """
        
        super(Flux_Kernels, self).__init__()
        
        # Extracting the spatial dimension and initial condition of the problem
        # and store the initial condition value u0
        self.Nx = u0.size(0)
        self.Ny = u0.size(1)
        self.u0 = u0
        
        # Set the device where training of the model takes place
        self.device = cfg.device
        
        # Variables that act as switch to use different types of boundary
        # condition
        # Each variable consists of boolean values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries (zero neumann_val)
        self.dirichlet_bool = cfg.dirichlet_bool[var_idx]
        self.neumann_bool = cfg.neumann_bool[var_idx]
        self.cauchy_bool = cfg.cauchy_bool[var_idx]
        
        # Variables that store the values of the boundary condition of each type
        # Values = 0 if not used, otherwise specify in the configuration file
        # Each variable consists of real values at all 2D domain boundaries:
        # [left (x = 0), right (x = Nx), top (y = 0), bottom (y = Ny)]
        # For 1D, only the first two values matter, set the last two values to
        # be no-flux boundaries
        if torch.is_tensor(cfg.dirichlet_val[var_idx]):
            self.dirichlet_val = cfg.dirichlet_val[var_idx].to(cfg.device)
        else:
            self.dirichlet_val = torch.tensor(cfg.dirichlet_val[var_idx]).to(cfg.device)
        
        if torch.is_tensor(cfg.neumann_val[var_idx]):
            self.neumann_val = cfg.neumann_val[var_idx].to(cfg.device)
        else:
            self.neumann_val = torch.tensor(cfg.neumann_val[var_idx]).to(cfg.device)
        
        # For Cauchy BC, the initial Cauchy value is set to be the initial
        # condition at each corresponding domain boundary, and will be updated
        # through time
        self.cauchy_val = []
        self.cauchy_val.append(u0[0, :].to(cfg.device))
        self.cauchy_val.append(u0[-1, :].to(cfg.device))
        self.cauchy_val.append(u0[:, 0].to(cfg.device))
        self.cauchy_val.append(u0[:, -1].to(cfg.device))
        
        # Set the Cauchy BC multiplier (to be multiplied with the gradient of
        # the unknown variable and the diffusion coefficient)
        if torch.is_tensor(cfg.cauchy_mult[var_idx]):
            self.cauchy_mult = cfg.cauchy_mult[var_idx].to(cfg.device)
        else:
            self.cauchy_mult = torch.tensor(cfg.cauchy_mult[var_idx]).to(cfg.device)
        
        # If numerical stencil is to be learned, initialize to +1 and -1 with
        # a standard deviation of 0.1 each, otherwise set it to fixed values
        self.learn_stencil = cfg.learn_stencil[var_idx]
        if self.learn_stencil:
            self.stencil = torch.tensor(
                [torch.normal(torch.tensor([1.0]), torch.tensor([0.1])),
                 torch.normal(torch.tensor([-1.0]), torch.tensor([0.1]))],
                dtype = torch.float)
            self.stencil = torch.nn.Parameter(self.stencil)
        else:
            self.stencil = torch.tensor([1.0, -1.0])
        
        # Extract the diffusion coefficient scalar value and set to be learnable
        # if desired
        if torch.is_tensor(cfg.D_eff[var_idx]):
            self.D_eff = cfg.D_eff[var_idx].to(cfg.device)
        else:
            self.D_eff = torch.tensor(cfg.D_eff[var_idx]).to(cfg.device)
        if cfg.learn_coeff[var_idx]:
            self.D_eff = torch.nn.Parameter(torch.tensor([self.D_eff],dtype=torch.float))
            
        # Extract the boolean value to determine whether the diffusion coefficient
        # is a function of the unknown variable
        self.coeff_func = cfg.coeff_func[var_idx]
        
        # Extract value of the normalizing constant to be applied to the output
        # of the NN that predicts the diffusion coefficient function
        if torch.is_tensor(cfg.p_exp_flux[var_idx]):
            self.p_exp = cfg.p_exp_flux[var_idx].to(cfg.device)
        else:
            self.p_exp = torch.tensor(cfg.p_exp_flux[var_idx]).to(cfg.device)
        
        # Initialize a NN to predict diffusion coefficient as a function of
        # the unknown variable if necessary
        if self.coeff_func:
            self.coeff_nn = Coeff_NN(cfg.num_layers_flux[var_idx], 
                                     cfg.num_nodes_flux[var_idx],
                                     len(cfg.flux_couple_idx[var_idx])).to(cfg.device)
            self.p_exp = torch.nn.Parameter(torch.tensor([self.p_exp],dtype=torch.float))

    
    def forward(self, u_main, u_coupled, t):
        
        """
        The forward function calculates the integrated flux between each control
        volume and its neighbors
        
        Inputs:
            u_main      : the unknown variable to be used to calculate the flux
                            indexed by flux_calc_idx[var_idx]
                            dim: [1, Nx, Ny]
                            
            u_coupled   : all necessary unknown variables required to calculate
                          the diffusion coeffient as a function, indexed by
                          flux_couple_idx[var_idx]
                          dim: [num_features, Nx, Ny]
                          
            t           : time (scalar value, taken from the ODE solver)
            
        Output:
            flux        : the integrated flux for all control volumes
                            dim: [Nx, Ny]
            
        """
        
        # Reshape the input dimension for the coeff_nn model into [Nx, Ny, num_features]
        u_coupled = u_coupled.permute(1,2,0)
        
        # Calculate the flux multiplier (diffusion coefficient function) if set
        # to be a function, otherwise set as tensor of ones
        if self.coeff_func:
            flux_mult = self.coeff_nn(u_coupled).squeeze(2) * 10**self.p_exp
        else:
            flux_mult = torch.ones(self.Nx, self.Ny)
           
        flux_mult = flux_mult.to(self.device)
        
        # Squeeze the u_main dimension into [Nx, Ny]
        u_main = u_main.squeeze(0)
        
        # Left Boundary Condition
        if self.dirichlet_bool[0]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            left_bound_flux = (self.stencil[0]*self.dirichlet_val[0] +
                               self.stencil[1]*u_main[0, :]).unsqueeze(0) \
                                * self.D_eff * flux_mult[0,:]
            
        elif self.neumann_bool[0]:
            # If Neumann, set the Neumann value as the flux at the boundary
            left_bound_flux = torch.cat(self.Ny * [self.neumann_val[0].unsqueeze(0)]).unsqueeze(0)
            
        elif self.cauchy_bool[0]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[0] = self.u0[0, :].to(self.device)
            else:
                self.cauchy_val[0] = (u_main[0, :]
                        - self.cauchy_val[0]) * self.cauchy_mult * self.D_eff
            # Calculate the flux at the boundary using the updated Cauchy value
            left_bound_flux = (self.stencil[0]*self.cauchy_val[0] +
                               self.stencil[1]*u_main[0, :]).unsqueeze(0) \
                                * self.D_eff * flux_mult[0,:]
        
        # Calculate the fluxes of each control volume with its left neighboring cell
        left_neighbors = (self.stencil[0]*u_main[:-1, :] + self.stencil[1]*u_main[1:, :]) \
                            * self.D_eff * flux_mult[1:, :]
        # Concatenate the left boundary fluxes with the left neighbors fluxes
        left_flux = torch.cat((left_bound_flux, left_neighbors))
        
        
        # Right Boundary Condition
        if self.dirichlet_bool[1]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            right_bound_flux = (self.stencil[0]*self.dirichlet_val[1] +
                                self.stencil[1]*u_main[-1, :]).unsqueeze(0) \
                                * self.D_eff * flux_mult[-1, :]
            
        elif self.neumann_bool[1]:
            # If Neumann, set the Neumann value as the flux at the boundary
            right_bound_flux = torch.cat(self.Ny * [self.neumann_val[1].unsqueeze(0)]).unsqueeze(0)
            
        elif self.cauchy_bool[1]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[1] = self.u0[-1, :].to(self.device)
            else:
                self.cauchy_val[1] = (u_main[-1, :]
                        - self.cauchy_val[1]) * self.cauchy_mult * self.D_eff
            # Calculate the flux at the boundary using the updated Cauchy value
            right_bound_flux = (self.stencil[0]*self.cauchy_val[1] +
                                self.stencil[1]*u_main[-1, :]).unsqueeze(0) \
                                * self.D_eff * flux_mult[-1,:]
        
        # Calculate the fluxes of each control volume with its right neighboring cell
        right_neighbors = (self.stencil[0]*u_main[1:, :] + self.stencil[1]*u_main[:-1, :]) \
                            * self.D_eff * flux_mult[:-1, :]
        # Concatenate the right neighbors fluxes with the right boundary fluxes
        right_flux = torch.cat((right_neighbors, right_bound_flux))
        
        
        # Top Boundary Condition
        if self.dirichlet_bool[2]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            top_bound_flux = (self.stencil[0]*self.dirichlet_val[2] +
                              self.stencil[1]*u_main[:, 0]).unsqueeze(1) \
                                * self.D_eff * flux_mult[:, 0]
            
        elif self.neumann_bool[2]:
            # If Neumann, set the Neumann value as the flux at the boundary
            top_bound_flux = torch.cat(self.Nx * [self.neumann_val[2].unsqueeze(0)]).unsqueeze(1)
            
        elif self.cauchy_bool[2]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[2] = self.u0[:, 0].to(self.device)
            else:
                self.cauchy_val[2] = (u_main[:, 0]
                        - self.cauchy_val[2]) * self.cauchy_mult * self.D_eff
            # Calculate the flux at the boundary using the updated Cauchy value
            top_bound_flux = (self.stencil[0]*self.cauchy_val[2] +
                              self.stencil[1]*u_main[:, 0]).unsqueeze(1) \
                                * self.D_eff * flux_mult[:, 0]
        
        # Calculate the fluxes of each control volume with its top neighboring cell
        top_neighbors = (self.stencil[0]*u_main[:, :-1] + self.stencil[1]*u_main[:, 1:]) \
                        * self.D_eff * flux_mult[:, 1:]
        # Concatenate the top boundary fluxes with the top neighbors fluxes
        top_flux = torch.cat((top_bound_flux, top_neighbors), dim=1)
        
        
        # Bottom Boundary Condition
        if self.dirichlet_bool[3]:
            # If Dirichlet, calculate the flux at the boundary using the
            # Dirichlet value as a constant
            bottom_bound_flux = (self.stencil[0]*self.dirichlet_val[3] +
                                 self.stencil[1]*u_main[:, -1]).unsqueeze(1) \
                                    * self.D_eff * flux_mult[:, -1]
            
        elif self.neumann_bool[3]:
            # If Neumann, set the Neumann value as the flux at the boundary
            bottom_bound_flux = torch.cat(self.Nx * [self.neumann_val[3].unsqueeze(0)]).unsqueeze(1)
            
        elif self.cauchy_bool[3]:
            # If Cauchy, first set the value to be equal to the initial condition
            # at t = 0.0, otherwise update the value according to the previous
            # time step value
            if t == 0.0:
                self.cauchy_val[3] = self.u0[:, -1].to(self.device)
            else:
                self.cauchy_val[3] = (u_main[:, -1]
                        - self.cauchy_val[3]) * self.cauchy_mult * self.D_eff
            # Calculate the flux at the boundary using the updated Cauchy value
            bottom_bound_flux = (self.stencil[0]*self.cauchy_val[3] +
                                 self.stencil[1]*u_main[:, -1]).unsqueeze(1) \
                                    * self.D_eff * flux_mult[:, -1]
        
        # Calculate the fluxes of each control volume with its bottom neighboring cell
        bottom_neighbors = (self.stencil[0]*u_main[:, 1:] + self.stencil[1]*u_main[:, :-1]) \
                            * self.D_eff * flux_mult[:, :-1]
        # Concatenate the bottom neighbors fluxes with the bottom boundary fluxes
        bottom_flux = torch.cat((bottom_neighbors, bottom_bound_flux), dim=1)
        
        # Integrate all fluxes at all control volume boundaries
        flux = left_flux + right_flux + top_flux + bottom_flux
            
        return flux

   
class Coeff_NN(torch.nn.Module):
    """
    The class Coeff_NN constructs a feedforward NN required for calculation
    of diffusion coefficient as a function of u
    It will be called in the Flux_Kernels constructor if the cfg.coeff_func is
    set to be True
    """
    
    def __init__(self, num_layers, num_nodes, num_vars):
        
        """
        Constructor
        
        Inputs:
            num_layers  : number of hidden layers (excluding output layer)
            num_nodes   : number of hidden nodes in each hidden layer
            num_vars    : number of features used as inputs
            
        """
        
        super(Coeff_NN, self).__init__()
        
        # Initialize the layer as an empty list
        layer = []
        
        # Add sequential layers as many as the specified num_layers, append
        # to the layer list, including the output layer (hence the +1 in the 
        # iteration range)
        for i in range(num_layers+1):
            
            #Specify number of input and output features for each layer
            in_features = num_nodes
            out_features = num_nodes
            
            # If it is the first hidden layer, set the number of input features
            # to be = num_vars
            if i == 0:
                in_features = num_vars
            # If it is the output layer, set the number of output features to be = 1
            elif i == num_layers:
                out_features = 1
            
            # Create sequential layer, if output layer use sigmoid activation function
            if i < num_layers:
                layer.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.Tanh()))
            else:
                layer.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.Sigmoid()))
                
        # Convert the list into a sequential module
        self.layers = torch.nn.Sequential(*layer)

    def forward(self, input):
        
        """
        The forward function calculates the approximation of diffusion coefficient
        function using the specified input values
        
        Input:
            input   : input for the function, all u that is required to calculate
                        the diffusion coefficient function (could be coupled with
                        other variables), dim: [Nx, Ny, num_features]
            
        Output:
            output      : the approximation of the diffusion coefficient function
            
        """
        
        output = self.layers(input)
        
        return output