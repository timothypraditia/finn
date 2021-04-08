# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Evaluate class that constructs the evaluation object
for the model and defines all functions required for testing and postprocessing
"""

import torch
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Evaluate:
    
    def __init__(self, x_pred, t, data, physmodel, core_num, x_data=None, x_phys=None,
                 breakthrough=False, profile=False):
        """
        Constructor
        
        Inputs:
            x_pred          : the spatial coordinates of the model prediction
                                (irrelevant for core sample #1 and #2)
            t               : time (a torch.tensor array containing all values of time steps
                                in which the output of the model will be calculated and
                                recorded)
            data            : test data
            physmodel       : prediction of the calibrated physical model
            core_num        : the core sample number (string: '1', '2', or '2B')
            x_data          : the spatial coordinates of the data (irrelevant for
                                core sample #1 and #2)
            x_phys          : the spatial coordinates of the physical model prediction
                                (irrelevant for core sample #1 and #2)
            breakthrough    : a Boolean value to set the postprocessing to
                                calculate the breakthrough curve
            profile         : a Boolean value to set the postprocessing to
                                calculate the total concentration profile

        """
        
        # Store data in the constructed object
        self.x_pred = x_pred
        if x_data is not None:
            self.x_data = x_data
        else:
            self.x_data = x_pred
        if x_phys is not None:
            self.x_phys = x_phys
        else:
            self.x_phys = x_pred
        self.x_phys = x_phys
        self.t = t
        self.data = data
        self.physmodel = physmodel
        self.core_num = core_num
        self.breakthrough = breakthrough
        self.profile = profile
        
    
    def evaluate(self, params, model, u0):
        """
        This function is the main function for postprocessing. It calculates
        the test prediction, compare it with the test data, and plot the results
        
        Inputs:
            params  : the configuration object containing the model settings
            model   : the trained model
            u0      : initial condition values, dim: [num_features, Nx, Ny]
        
        """
        
        # Set the model to evaluation mode
        model.eval()
        
        # Calculate prediction using the trained model
        ode_pred = odeint(model, u0.to(params.device),
                                self.t.to(params.device), rtol=1e-5, atol=1e-6)
        
        # Set the x-axis value for plotting. If breakthrough curve, the x-axis
        # represents the time, and if concentration profile, the x-axis represents
        # the depth of the core sample
        if self.breakthrough:
            self.calculate_breakthrough(model, ode_pred)
            x_data_plot = self.t
            x_pred_plot = self.t
            x_phys_plot = self.t
            
        elif self.profile:
            self.calculate_profile(ode_pred)
            x_data_plot = self.x_data
            x_pred_plot = self.x_pred
            x_phys_plot = self.x_phys
          
        # Calculate normalized MSE (since the breakthrough curve value is very small)
        self.norm_mse_pred = torch.mean((self.norm_data.to(params.device) - self.norm_pred)**2)
        self.norm_mse_physmodel = torch.mean((self.norm_data - self.norm_physmodel)**2)
        
        # Save prediction of dissolved and total concentration in .csv files if required
        if params.save_model:
            pd.DataFrame(np.squeeze(ode_pred[:,0].squeeze().cpu().detach().numpy())).to_csv(
                params.model_path + '\\core' + self.core_num + '_cdiss_prediction.csv', sep = "\t", float_format = '%.4f')
            pd.DataFrame(np.squeeze(ode_pred[:,1].squeeze().cpu().detach().numpy())).to_csv(
                params.model_path + '\\core' + self.core_num + '_ctot_prediction.csv', sep = "\t", float_format = '%.4f')
        
        # Plot the results
        self.plot(params, x_data_plot, x_pred_plot, x_phys_plot)
    
    
    def calculate_breakthrough(self, model, ode_pred):
        """
        This function extracts the breakthrough curve prediction.
        
        Inputs:
            model       : the trained model
            ode_pred    : the full field solution prediction
            
        """
        
        # Calculate the predicted breakthrough curve with its absolute values
        # for plotting
        cauchy_mult = model.flux_modules[0].cauchy_mult * model.flux_modules[0].D_eff
        self.plot_pred = ((ode_pred[:,0,-2] - ode_pred[:,0,-1]) * cauchy_mult).squeeze()
        
        # Calculate the normalized data, and prediction of both FINN and the 
        # physical model for MSE calculation
        self.norm_pred, self.norm_data = self.normalize(self.plot_pred, self.data)
        self.norm_physmodel, _ = self.normalize(self.physmodel, self.data)
    
    
    def calculate_profile(self, ode_pred):
        """
        This function extracts the total concentration profile prediction.
        
        Input:
            ode_pred : the full field solution prediction
            
        """
        
        # Extract the predicted total concentration profile with its absolute
        # values for plotting
        self.plot_pred = ode_pred[-1,1].squeeze()
        
        # Interpolate the prediction at the spatial coordinates of the real
        # experimental data (for MSE calculation so that they are calculated
        # at the same spatial locations)
        if self.x_data is not None:
            f = interp1d(self.x_pred, self.plot_pred.cpu().detach())
            pred = torch.FloatTensor(f(self.x_data))
        else:
            pred = self.plot_pred
        
        # Calculate the normalized data, and prediction of both FINN and the 
        # physical model for MSE calculation
        self.norm_pred, self.norm_data = self.normalize(pred, self.data)
        self.norm_physmodel, _ = self.normalize(self.physmodel, self.data)
        
    
    def normalize(self, pred, data):
        """
        This function normalizes the data and prediction of the model.
        
        Inputs:
            pred    : prediction of the model
            data    : test data

        """
        
        # Normalize data and prediction relative to the minimum and maximum value
        # of the data
        norm_data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        norm_pred = (pred - torch.min(data)) / (torch.max(data) - torch.min(data))
        
        return norm_pred, norm_data

        
    def plot(self, params, x_data_plot, x_pred_plot, x_phys_plot):    
        """
        This function plots the prediction results, compared with the data
        and the physical model prediction.
        
        Inputs:
            params          : the configuration object containing the model settings
            x_data_plot     : x-axis values for the data
            x_pred_plot     : x-axis values for the FINN prediction
            x_phys_plot     : x-axis values for the physical model prediction

        """
        
        plt.figure()
        plt.scatter(x_data_plot, self.data, label='Data')
        plt.plot(x_pred_plot, self.plot_pred.cpu().detach(), label ='NN')
        plt.plot(x_phys_plot, self.physmodel, label='Physical Model')
        plt.legend(fontsize=16)
        
        if self.breakthrough:
            caption = 'Breakthrough Curve of Core #' + self.core_num
            x_label = 'time [days]'
            y_label = 'Tailwater concentration [mg/L]'
        elif self.profile:
            caption = r'Total Concentration Profile of Core #' + self.core_num + \
                ' at $t = 48.88$ days'
            x_label = 'Depth [m]'
            y_label = 'TCE Concentration [mg/L]'
        
        plt.title(caption, fontsize=16)
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.tight_layout()
        
        if params.save_model:
            plt.savefig(params.model_path + "\\" + params.model_name +
                        "_core" + self.core_num + ".png")
    
    
    
        