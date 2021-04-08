# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Evaluate class that constructs the evaluation object
for the model and defines all functions required for testing and postprocessing
"""

import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class Evaluate:
    
    def __init__(self, x, t, data):
        """
        Constructor
        
        Inputs:
            x       : the spatial coordinates of the data and model prediction
            t       : time (a torch.tensor array containing all values of time steps
                        in which the output of the model will be calculated and
                        recorded)
            data    : test data
            
        """
        
        self.x = x
        self.t = t
        self.data = data
        
        
    def evaluate(self, cfg, model, u0, unseen):
        """
        This function is the main function for postprocessing. It calculates
        the test prediction, compare it with the test data, and plot the results
        
        Inputs:
            params  : the configuration object containing the model settings
            model   : the trained model
            u0      : initial condition values, dim: [num_features, Nx, Ny]
            unseen  : a Boolean value to determine whether this object is the
                        extrapolated training case or the unseen test case
        
        """
        
        # Set the model to evaluation mode
        model.eval()
        
        # Calculate prediction using the trained model
        self.ode_pred = odeint(model, u0.to(cfg.device),
                                self.t.to(cfg.device), rtol=1e-5, atol=1e-6)
        
        # Calculate MSE of the FINN prediction
        self.mse_test = torch.mean((self.data.to(cfg.device) - self.ode_pred)**2).item()
        
        # Extract the breakthrough curve prediction
        self.pred_breakthrough = self.ode_pred[:,0,-1].squeeze()
        
        # Plot breakthrough curve if this object is the extrapolated training case
        if not unseen:
            self.plot_breakthrough_sep(cfg)
            
        # Plot the full field solution of both dissolved and total concentration
        self.plot_full_field(cfg, self.ode_pred[:,0].squeeze(), self.data[:,0].squeeze(), True)
        self.plot_full_field(cfg, self.ode_pred[:,1].squeeze(), self.data[:,1].squeeze(), False)
        
        
        
    def plot_breakthrough_sep(self, cfg):    
        """
        This function plots the predicted breakthrough curve in comparison
        to the data
        
        Input:
            cfg : the configuration object containing the model settings
                
        """
        
        plt.figure()
        plt.plot(self.t, self.data[:,0,-1].squeeze(), label='Data')
        
        # Determine index of the prediction to be plotted (so that the scatter
        # plot marker is sparse and visualization is better)
        train_plot_idx = torch.arange(1, 501, 50)
        test_plot_idx = torch.arange(501, 2001, 50)
        
        # Plot the predicted breakthrough curve, with different color for
        # training and the extrapolation
        plt.scatter(self.t[train_plot_idx], self.pred_breakthrough[train_plot_idx].cpu().detach(),
                    label='Training', color='green', marker='x')
        plt.scatter(self.t[test_plot_idx], self.pred_breakthrough[test_plot_idx].cpu().detach(),
                    label='Testing', color='red', marker='x')
        plt.legend(fontsize=16)
        
        # Plot a black vertical line as separator between training and extrapolation
        sep_t = torch.cat(2 * [torch.tensor(self.t[501]).unsqueeze(0)])
        sep_y = torch.tensor([0.0, 1.1*torch.max(self.data[:,0,-1])])
        plt.plot(sep_t, sep_y, color='black')
        
        # Determine caption depending on which isotherm is being used
        if cfg.linear:
            caption = 'Linear'
        elif cfg.freundlich:
            caption = 'Freundlich'
        elif cfg.langmuir:
            caption = 'Langmuir'
        plt.title('Breakthrough Curve (' + caption + ' Sorption)', fontsize=16)
        plt.xlabel('time [days]', fontsize=16)
        plt.ylabel('Tailwater concentration [mg/L]', fontsize=16)
        plt.tight_layout()
        
        # Save plot if required
        if cfg.save_model:
            plt.savefig(cfg.model_path + "\\" + cfg.model_name + "_breakthrough_curve.png")
    
    
    def plot_full_field(self, cfg, pred, data, diss):
        """
        This function plots the full field solution of the model prediction in
        comparison to the test data
        
        Inputs:
            cfg     : the configuration object containing the model settings
            pred    : the model prediction
            data    : test data
            diss    : a Boolean value to determine whether this is a plot for
                        the dissolved or total concentration
                
        """
        
        plt.figure(figsize=(10.0,5.0))
        plt.subplot(121)
        plt.pcolormesh(self.x, self.t, data)
        if diss:
            caption = 'Dissolved'
            save_name = 'diss'
        else:
            caption = 'Total'
            save_name = 'tot'
        plt.title(caption + ' Concentration Data', fontsize=16)
        plt.xlabel('Depth [m]', fontsize=16)
        plt.ylabel('time [days]', fontsize=16)
        plt.colorbar()
        plt.clim([0, torch.max(data)])
        
        
        plt.subplot(122)
        plt.pcolormesh(self.x, self.t, pred.cpu().detach())
        plt.title(caption + ' Concentration Prediction', fontsize=16)
        plt.xlabel('Depth [m]', fontsize=16)
        plt.ylabel('time [days]', fontsize=16)
        plt.colorbar()
        plt.clim([0, torch.max(data)])
        
        plt.tight_layout()
        
        if cfg.save_model:
            plt.savefig(cfg.model_path + "\\" + cfg.model_name + "_c_" + save_name + ".png")