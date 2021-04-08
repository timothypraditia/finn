# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script is the main file to train and test FINN with synthetic dataset
"""

import torch
import os
import sys
sys.path.insert(0, os.path.abspath("../../kernels"))
import net_model
import syn02_init as init
import syn03_training as train
import syn04_evaluate as test
import pandas as pd
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Initialization of configurations and set modules for different core samples
cfg = init.Initialize()


# Initialize the model to be trained
u0 = torch.zeros(cfg.num_vars, cfg.Nx, 1)
model = net_model.Net_Model(u0, cfg)


# Load training data with the original dimension of [Nx, Nt]
# Reshape into [Nt, 1, Nx, Ny], with Ny = 1
if cfg.linear:
    folder = 'data_linear'
elif cfg.freundlich:
    folder = 'data_freundlich'
elif cfg.langmuir:
    folder = 'data_langmuir'
    
c_diss = pd.read_csv(folder + '\\c_diss.csv', sep='\t', header=None)
c_diss = torch.tensor(np.array(c_diss)).unsqueeze(1)
c_diss = c_diss.permute(2,0,1).unsqueeze(1)

c_tot = pd.read_csv(folder + '\\c_tot.csv', sep='\t', header=None)
c_tot = torch.tensor(np.array(c_tot)).unsqueeze(1)
c_tot = c_tot.permute(2,0,1).unsqueeze(1)


# Concatenate dissolved and total concentration data along the second dimension
# (dim=1 in Python)
train_data = torch.cat((c_diss,c_tot), dim=1)


# Train the model
x = torch.linspace(0.0, cfg.X, cfg.Nx)
t = torch.linspace(0.0, cfg.T, cfg.Nt)
trainer = train.Training(model, cfg)
trainer.model_train(u0, t[:501], train_data[:501])


# Load testing data with the original dimension of [Nx, Nt]
# Reshape into [Nt, 1, Nx, Ny], with Ny = 1
c_diss_test = pd.read_csv(folder + '\\c_diss_test.csv', sep='\t', header=None)
c_diss_test = torch.tensor(np.array(c_diss_test)).unsqueeze(1)
c_diss_test = c_diss_test.permute(2,0,1).unsqueeze(1)

c_tot_test = pd.read_csv(folder + '\\c_tot_test.csv', sep='\t', header=None)
c_tot_test = torch.tensor(np.array(c_tot_test)).unsqueeze(1)
c_tot_test = c_tot_test.permute(2,0,1).unsqueeze(1)


# Concatenate dissolved and total concentration data along the second dimension
# (dim=1 in Python)
test_data = torch.cat((c_diss_test, c_tot_test),dim=1)


# Evaluate the trained model with extrapolation of the train dataset
extrapolate_train = test.Evaluate(x, t, train_data)
extrapolate_train.evaluate(cfg, model, u0, False)


# Evaluate the trained model with unseen test dataset
model_test = model
model_test.flux_modules[0].dirichlet_val[0] = 0.7
model_test.flux_modules[1].dirichlet_val[0] = 0.7
unseen_test = test.Evaluate(x, t, test_data)
unseen_test.evaluate(cfg, model_test, u0, True)


# Save the configuration and performance of the trained and tested model
# and hyperparameter settings into the log file if required
if cfg.save_model:
    # Open the configuration and performance log file
    with open(cfg.model_path + '\\' + cfg.model_name + '_cfg_and_performance.txt', 'r') as f:
        cfg_file = f.read()
        
    output_string = cfg_file + "\n\n# Testing Performance\n\n"
    
    output_string += "EXTRAPOLATE TRAIN MSE = " + \
                      str(extrapolate_train.mse_test) + "\n"
    output_string += "UNSEEN TEST MSE = " + \
                      str(unseen_test.mse_test) + "\n"
    
    # Save the updated performance metrics into the file
    with open(cfg.model_path + '\\' + cfg.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
        _text_file.write(output_string)
        
    
    # Extract the training and validation loss values from the latest
    # checkpoint (i.e. with the best performance)
    train_loss_save = trainer.checkpoint['loss_train'][-1]
    
    
    # Define the hparam_dict that contains the hyperparameters values
    hparam_dict = {'flux_layers': cfg.flux_layers,
                    'state_layers': cfg.state_layers,
                    'flux_nodes': cfg.flux_nodes,
                    'state_nodes': cfg.state_nodes,
                    'learning_rate': cfg.learning_rate,
                    'error_mult': cfg.error_mult,
                    'breakthrough_mult': cfg.breakthrough_mult,
                    'profile_mult': cfg.profile_mult,
                    'phys_mult': cfg.phys_mult,
                    'epochs': cfg.epochs,
                    'lbfgs': cfg.lbfgs_optim,
                    'train_breakthrough': cfg.train_breakthrough,
                    'linear': cfg.linear,
                    'freundlich': cfg.freundlich,
                    'langmuir': cfg.langmuir
                    }
    
    
    # Define the metric_dict that contains the performance metrics values
    metric_dict = {'hparam/train_loss': train_loss_save,
                    'hparam/mse_extrapolate': extrapolate_train.mse_test,
                    'hparam/mse_unseen': unseen_test.mse_test}
    
    # Write the hparams_dict and metric_dict to the tensorboard log file
    trainer.tb.add_hparams(hparam_dict, metric_dict)