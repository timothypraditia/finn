# -*- coding: utf-8 -*-
"""
Created on March 2021
@author: Timothy Praditia

This script contains the Training class that constructs the training object
for the model and defines all functions required for training
"""

import torch
from torchdiffeq import odeint
import numpy as np
import time
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Training:
    
    def __init__(self, model, cfg):
        """
        Constructor
        
        Inputs:
            model   : the model object constructed using the Net_Model class
            cfg     : the configuration object containing the model settings
            
        """
        
        self.cfg = cfg
        
        # Send model to the corresponding device (important when using GPU)
        self.model = model.to(self.cfg.device)
        
        # Choose between ADAM or LBFGS as the optimizer
        # LBFGS theoretically should work better compared to ADAM, but the
        # memory requirement and computation time is also higher
        
        if self.cfg.lbfgs_optim:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr = self.cfg.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = self.cfg.learning_rate)
        
        # Initialize the initial epoch value, an empty list to store the training
        # loss values, and set initial best loss value as infinity, to be updated
        # after each iteration
        self.start_epoch = 0
        self.train_loss = []
        self.best_loss = np.infty
        
        # Define the filename to save and/or load the model
        self.model_save_file = self.cfg.model_path + "\\" + self.cfg.model_name + ".pt"
        
        # Create a Tensorboard summary writer instance in the log directory
        # The Tensorboard summary includes the training and validation loss,
        # as well as hyperparameters values to be compared with other models
        self.tb = SummaryWriter(self.cfg.log_path)
        
        
        # Load the model if this instance is a training continuation from a
        # previous checkpoint
        if self.cfg.continue_training:
            print('Restoring model (that is the network\'s weights) from file...')
            print()
            
            # Load the latest checkpoint
            self.checkpoint = torch.load(self.model_save_file)
            
            # Load the model state_dict (all the network parameters) and send
            # the model to the corresponding device
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.model.to(self.cfg.device)
            
            # Load the optimizer state dict (important because ADAM and LBFGS 
            # requires past states, e.g. momentum information and approximate
            # Hessian)
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.cfg.device)
            
            # Load the epoch and loss values from the previous training up until
            # the checkpoint to enable complete history of the training
            self.start_epoch = self.checkpoint['epoch']
            self.train_loss = self.checkpoint['loss_train']
            
            # Store the loss values in the Tensorboard log file
            for epoch in range(self.start_epoch):
                self.tb.add_scalar('training_loss', self.train_loss[epoch], epoch)
        
        # Calculate the "true" retardation factor based on the specified correlation
        # for plotting purposes to be compared with the approximation
        u = torch.linspace(0.01, 1.00, 100)
        self.retardation_linear = torch.cat(len(u) *
                    [torch.tensor(1 + (1-self.cfg.por)/self.cfg.por * self.cfg.rho_s
                                  * self.cfg.Kd).unsqueeze(0)])
        self.retardation_freundlich = 1 + (1-self.cfg.por)/self.cfg.por * self.cfg.rho_s \
                * self.cfg.Kf * self.cfg.nf * (u + 1e-6)**(self.cfg.nf - 1)
        self.retardation_langmuir = 1 + (1-self.cfg.por)/self.cfg.por * self.cfg.rho_s \
                * self.cfg.smax * self.cfg.Kl / ((u + self.cfg.Kl)**2)
            
            
    def model_train(self, u0, t, data):
        """
        This function trains the model
        
        Inputs:
            u0      : initial condition, dim: [num_features, Nx, Ny]
            t       : time (a torch.tensor array containing all values of time steps
                        in which the output of the model will be calculated and
                        recorded)
            data    : training data (full field solution with the same shape as u0),
                        dim: [num_features, Nx, Ny]
            
        """
        
        # Set the number of threads for this program to one
        torch.set_num_threads(1)
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # The closure function is necessary for LBFGS optimizer, because
        # it requires multiple function evaluations
        # The closure function returns the loss value
        def closure():
            
            # Set the model to training mode
            self.model.train()
            
            # Reset the gradient buffer (set to 0)
            self.optimizer.zero_grad()
            
            # Calculate the model prediction (full field solution)
            ode_pred = odeint(self.model, u0.to(self.cfg.device),
                              t.to(self.cfg.device), rtol=1e-5, atol=1e-6)
            
            # If trained only using breakthrough curve data calculate the
            # predicted breakthrough, else use the whole ode_pred values
            if self.cfg.train_breakthrough:
                # Extract the breakthrough curve prediction and data
                cauchy_mult = self.model.flux_modules[0].cauchy_mult * self.model.flux_modules[0].D_eff
                pred_breakthrough = ((ode_pred[:,0,-2] - ode_pred[:,0,-1]) * cauchy_mult).squeeze()
                data_breakthrough = data[:,0,-1].squeeze()
                
                # Calculate loss based on the breakthrough curve prediction and data
                loss = self.cfg.breakthrough_mult * torch.sum((data_breakthrough.to(self.cfg.device)
                                                    - pred_breakthrough)**2)
                
                # Extract the total concentration profile at t_end
                pred_profile = ode_pred[-1,1].squeeze()
                data_profile = data[-1,1].squeeze()
                
                # Calculate the loss based on the concentration profile prediction and data
                loss += self.cfg.profile_mult * torch.sum((data_profile.to(self.cfg.device)
                                                           - pred_profile)**2)
                
            else:
                # Calculate loss using the sum squared error metric
                loss = self.cfg.error_mult * torch.sum((data.to(self.cfg.device)
                                                    - ode_pred)**2)
            
            
            # Extract the predicted retardation factor function for physical
            # regularization
            u = torch.linspace(0.0, 1.0, 100).view(-1,1).to(self.cfg.device)
            ret_temp = self.model.flux_modules[0].coeff_nn(u)
            
            # Physical regularization: value of the retardation factor should
            # decrease with increasing concentration
            loss += self.cfg.phys_mult * torch.sum(
                torch.relu(ret_temp[:-1] - ret_temp[1:]))
            
            # Backpropagate to obtain gradient of model parameters
            loss.backward()
            
            return loss
        
        # Plot the predicted retardation factor as a function of dissolved
        # concentration and update at each training epoch
        fig, ax = plt.subplots()
        u = torch.linspace(0.01, 1.00, 100).view(-1,1).to(self.cfg.device)
        plt.plot(u.cpu(), self.retardation_linear, linestyle="--", label='Linear')
        plt.plot(u.cpu(), self.retardation_freundlich, linestyle="--", label='Freundlich')
        plt.plot(u.cpu(), self.retardation_langmuir, linestyle="--", label='Langmuir')
        ret_pred = 1 / self.model.flux_modules[0].coeff_nn(u) / 10**self.model.flux_modules[0].p_exp
        ax_pred, = ax.plot(u.cpu(), ret_pred.cpu().detach(), label='FINN')
        plt.title('Predicted Retardation Factor',fontsize=16)
        plt.xlabel(r'$c_{diss}$ [mg/L]',fontsize=16)
        plt.ylabel(r'$R$',fontsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        
        # Iterate until maximum epoch number is reached
        for epoch in range(self.start_epoch, self.cfg.epochs):
            # Start timer
            a = time.time()
            
            # Update the model parameters and record the loss value
            self.optimizer.step(closure)
            loss = closure()
            self.train_loss.append(loss.item())
            
            # If the training loss is lower than the best loss value,
            # update the best loss and save the model
            if self.train_loss[-1] < self.best_loss:
                self.best_loss = self.train_loss[-1]
                if self.cfg.save_model:
                    thread = Thread(target=self.save_model_to_file(
                        epoch))
                    thread.start()
                    
            # Write the loss values to the tensorboard log file
            self.tb.add_scalar('training_loss', self.train_loss[-1], epoch)
            
            # Stop the timer
            b = time.time()
            
            # Print out the epoch status
            print('Training: Epoch [%d/%d], Training Loss: %.4f, Runtime: %.4f secs'
                  %(epoch + 1, self.cfg.epochs, self.train_loss[-1], b - a))
            
            # Update the retardation factor plot
            ret_pred = 1 / self.model.flux_modules[0].coeff_nn(u) / 10**self.model.flux_modules[0].p_exp
            ax_pred.set_ydata(ret_pred.cpu().detach())
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.0001)
        
        # Load model from the latest saved checkpoint (i.e. with the lowest
        # training error)
        if self.cfg.save_model:
            self.checkpoint = torch.load(self.model_save_file)
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.model.to(self.cfg.device)
        
        # Plot the retardation factor and save if required
        ret_pred = 1 / self.model.flux_modules[0].coeff_nn(u) / 10**self.model.flux_modules[0].p_exp
        ax_pred.set_ydata(ret_pred.cpu().detach())
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.0001)
        if self.cfg.save_model:
            plt.savefig(self.cfg.model_path + "\\" + self.cfg.model_name + "_retardation.png")
        
        
    def save_model_to_file(self, epoch):
        """
        This function writes the model weights along with the network configuration
        and current performance to file
        
        Input:
            epoch : the current epoch number during training
            
        """
        
        # Save model weights, optimizer state_dict, and epoch status to file
        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'loss_train': self.train_loss}
        torch.save(state, self.model_save_file)
        
        # Write the training performance and the configuration of the model to 
        # a file
        with open('syn00_config.py', 'r') as f:
            cfg_file = f.read()
        
        output_string = cfg_file + "\n\n# Training Performance\n\n"
    
        output_string += "CURRENT_EPOCH = " + str(epoch+1) + "\n"
        output_string += "EPOCHS = " + str(self.cfg.epochs) + "\n"
        output_string += "CURRENT_TRAINING_ERROR = " + \
                         str(self.train_loss[-1]) + "\n"
        output_string += "LOWEST_TRAINING_ERROR = " + \
                         str(min(self.train_loss))
    
        # Save the configuration and current performance to file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
            _text_file.write(output_string)