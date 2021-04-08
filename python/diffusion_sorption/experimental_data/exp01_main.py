# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script is the main file to train and test FINN with experimental dataset
"""

import torch
import os
import sys
sys.path.insert(0, os.path.abspath("../../kernels"))
import net_model
import exp02_init as init
import exp03_set_module as set_module
import exp04_training as train
import exp05_evaluate as test
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Initialization of configurations and set modules for different core samples
params = init.Initialize()
core1_cfg = set_module.Set_Module('data_core1.xlsx', params)
core2_cfg = set_module.Set_Module('data_core2.xlsx', params)
core2b_cfg = set_module.Set_Module('data_core2_long.xlsx', params)


# Initialize the model to be trained using data from core sample #2
u0 = torch.zeros(params.num_vars,core2_cfg.Nx,1)
model_core2 = net_model.Net_Model(u0, core2_cfg)


# Read the core #2 data from the Excel file
data_core2 = pd.read_excel('data_core2.xlsx', index_col=None, header=None)
breakthrough_data_core2 = torch.FloatTensor(data_core2[1]) / 1000
breakthrough_time_core2 = torch.FloatTensor(data_core2[0])


# Train the model
trainer = train.Training(model_core2, params)
trainer.model_train(u0, breakthrough_time_core2, breakthrough_data_core2)


# Test the trained model with core sample #2
core2_physmodel = pd.read_excel('data_core2.xlsx', sheet_name=2, index_col=None, header=None)
core2_physmodel = torch.FloatTensor(core2_physmodel[1])/1000

core2_x_pred = torch.linspace(core2_cfg.X/(core2_cfg.Nx + 1), core2_cfg.X, core2_cfg.Nx)
core2_eval = test.Evaluate(core2_x_pred, breakthrough_time_core2, breakthrough_data_core2,
                            core2_physmodel, '2', breakthrough=True)
core2_eval.evaluate(params, model_core2, u0)


# Test the trained model with core sample #1
data_core1 = pd.read_excel('data_core1.xlsx', index_col=None, header=None)
breakthrough_data_core1 = torch.FloatTensor(data_core1[1]) / 1000
breakthrough_time_core1 = torch.FloatTensor(data_core1[0])

u0 = torch.zeros(params.num_vars,core1_cfg.Nx,1)
model_core1 = net_model.Net_Model(u0, core1_cfg)
model_core1.load_state_dict(trainer.model.state_dict())
model_core1.to(params.device)

core1_physmodel = pd.read_excel('data_core1.xlsx', sheet_name=2, index_col=None, header=None)
core1_physmodel = torch.FloatTensor(core1_physmodel[1])/1000

core1_x_pred = torch.linspace(core1_cfg.X/(core1_cfg.Nx + 1), core1_cfg.X, core1_cfg.Nx)
core1_eval = test.Evaluate(core1_x_pred, breakthrough_time_core1, breakthrough_data_core1,
                            core1_physmodel, '1', breakthrough=True)
core1_eval.evaluate(params, model_core1, u0)


# Test the trained model with core sample #2B
data_core2b = pd.read_excel('data_core2_long.xlsx', index_col=None, header=None)
profile_data_core2b = torch.FloatTensor(data_core2b[1]) / 1000
profile_x_core2b = torch.FloatTensor(data_core2b[0])
time_core2b = torch.linspace(0.0, core2b_cfg.T, 101)

u0 = torch.zeros(params.num_vars,core2b_cfg.Nx,1)
model_core2b = net_model.Net_Model(u0, core2b_cfg)
model_core2b.load_state_dict(trainer.model.state_dict())
model_core2b.to(params.device)

data_core2b_physmodel = pd.read_excel('data_core2_long.xlsx', sheet_name=2, index_col=None, header=None)
core2b_physmodel = torch.FloatTensor(data_core2b_physmodel[1])/1000

core2b_x_pred = torch.linspace(core2b_cfg.X/(core2b_cfg.Nx + 1), core2b_cfg.X, core2b_cfg.Nx)
core2b_eval = test.Evaluate(core2b_x_pred, time_core2b, profile_data_core2b[1:],
                            core2b_physmodel[1:], '2B', x_data=profile_x_core2b[1:], x_phys=profile_x_core2b[1:], profile=True)
core2b_eval.evaluate(params, model_core2b, u0)


# Save the configuration and performance of the trained and tested model
# and hyperparameter settings into the log file if required
if params.save_model:
    # Open the configuration and performance log file
    with open(params.model_path + '\\' + params.model_name + '_cfg_and_performance.txt', 'r') as f:
        cfg_file = f.read()
        
    output_string = cfg_file + "\n\n# Testing Performance\n\n"
    
    output_string += "CORE 2 MSE = " + \
                      str(core2_eval.norm_mse_pred.item()) + "\n"
    output_string += "CORE 2 MSE (PHYS MODEL) = " + \
                      str(core2_eval.norm_mse_physmodel.item()) + "\n"
    output_string += "CORE 1 MSE = " + \
                      str(core1_eval.norm_mse_pred.item()) + "\n"
    output_string += "CORE 1 MSE (PHYS MODEL) = " + \
                      str(core1_eval.norm_mse_physmodel.item()) + "\n"
    output_string += "CORE 2B MSE = " + \
                      str(core2b_eval.norm_mse_pred.item()) + "\n"
    output_string += "CORE 2B MSE (PHYS MODEL) = " + \
                      str(core2b_eval.norm_mse_physmodel.item()) + "\n"
    
    # Save the updated performance metrics into the file
    with open(params.model_path + '\\' + params.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
        _text_file.write(output_string)
        
    
    # Extract the training and validation loss values from the latest
    # checkpoint (i.e. with the best performance)
    train_loss_save = trainer.checkpoint['loss_train'][-1]
    
    
    # Define the hparam_dict that contains the hyperparameters values
    hparam_dict = {'flux_layers': params.flux_layers,
                    'state_layers': params.state_layers,
                    'flux_nodes': params.flux_nodes,
                    'state_nodes': params.state_nodes,
                    'learning_rate': params.learning_rate,
                    'error_mult': params.error_mult,
                    'phys_mult': params.phys_mult,
                    'epochs': params.epochs,
                    'lbfgs': params.lbfgs_optim}
    
    # Define the metric_dict that contains the performance metrics values
    metric_dict = {'hparam/train_loss': train_loss_save,
                    'hparam/mse_core2': core2_eval.norm_mse_pred.item(),
                    'hparam/mse_core1': core1_eval.norm_mse_pred.item(),
                    'hparam/mse_core2b': core2b_eval.norm_mse_pred.item()}
    
    # Write the hparams_dict and metric_dict to the tensorboard log file
    trainer.tb.add_hparams(hparam_dict, metric_dict)