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
import exp00_config as params

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
        self.phys_mult = params.phys_mult
        self.epochs = params.epochs
        self.lbfgs_optim = params.lbfgs_optim
        
        # SIMULATION-RELATED INPUTS
        self.num_vars = params.num_vars
        
    
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