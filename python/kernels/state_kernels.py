# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the State_Kernels class, which takes input of the integrated
fluxes calculated by the flux kernels, and u to calculate the reaction function
whenever necessary, and returns the du/dt approximation
"""

import torch


class State_Kernels(torch.nn.Module):
    
    def __init__(self, cfg, var_idx):
        
        """
        Constructor
        Inputs:
            cfg     : configuration object of the model setup, containing boundary
                        condition types, values, learnable parameter settings, etc.
            var_idx : index of the calculated variable (could be > 1 for coupled
                        systems)
        """
        
        super(State_Kernels, self).__init__()
        
        # Extract the boolean value to determine whether there is any reaction
        # that is a function of the unknown variable
        self.react_func = cfg.react_func[var_idx]
        
        # Extract value of the normalizing constant to be applied to the output
        # of the NN that predicts the reaction function
        if torch.is_tensor(cfg.p_exp_state[var_idx]):
            self.p_exp = cfg.p_exp_state[var_idx].to(cfg.device)
        else:
            self.p_exp = torch.tensor(cfg.p_exp_state[var_idx]).to(cfg.device)
        
        # Initialize a NN to predict reaction as a function of the unknown
        # variable if necessary
        if self.react_func:
            self.react_nn = React_NN(cfg.num_layers_state[var_idx],
                                     cfg.num_nodes_state[var_idx],
                                     len(cfg.state_couple_idx[var_idx])).to(cfg.device)
            self.p_exp = torch.nn.Parameter(torch.tensor([self.p_exp],dtype=torch.float))
        
    
    def forward(self, u_coupled, flux):
        
        """
        The forward function calculates du/dt
        
        Inputs:
            u_coupled   : all necessary unknown variables required to calculate
                            the reaction as a function, indexed by
                            state_couple_idx[var_idx], dim: [num_features, Nx, Ny]
                            
            flux        : the integrated flux calculated by the flux kernels,
                            dim: [Nx, Ny]
            
        Output:
            state       : du/dt, dim: [Nx, Ny]
            
        """
        
        # Reshape the input dimension for the react_nn model into [Nx, Ny, num_features]
        u_coupled = u_coupled.permute(1,2,0)
        
        # Calculate du/dt
        state = flux
        # If there is a reaction function, add it into the fluxes
        if self.react_func:
            state += self.react_nn(u_coupled).squeeze(2)
                
        return state
    

class React_NN(torch.nn.Module):
    """
    The class React_NN constructs a feedforward NN required for calculation
    of reaction term as a function of u
    It will be called in the State_Kernels constructor if the cfg.react_func is
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
        
        super(React_NN, self).__init__()
        
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
            
            # Create sequential layers with the hyperbolic tangent activation function
            layer.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.Tanh()))
        
        # Convert the list into a sequential module
        self.layers = torch.nn.Sequential(*layer)

    def forward(self, input):
        
        """
        The forward function calculates the approximation of the reaction
        function using the specified input values
        
        Input:
            input   : input for the function, all u that is required to calculate
                        the diffusion coefficient function (could be coupled with
                        other variables), dim: [Nx, Ny, num_features]
            
        Output:
            output      : the approximation of the reaction function
            
        """
        
        
        output = self.layers(input)
        
        return output