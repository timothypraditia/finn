# Finite Volume Neural Network
This is the code and data repository for the Finite Volume Neural Network method using the Python language.

## Requirement
To run the codes in this repository, the following packages are required:
- [torch](https://pytorch.org/get-started/locally/)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- [tensorboard](https://pytorch.org/docs/stable/tensorboard.html)

## Organization
This repository contains the implementation of the FINN method for the non-linear diffusion-sorption process.
The implementation is divided in two parts: synthetic and experimental data.

## Implementation
The implementation of the flux and state kernels, and the model class can be found in [this folder](kernels).
To configure and set up the FINN model, adjust the values in the configuration files.
The model is set up for 2-D problems, and therefore, for 1-D problems, use only the first 2 dimensions, then set the `neumann_val[2]` and `neumann_val[3]` to be 0.


## Synthetic Dataset
All the codes and dataset can be found [here](diffusion_sorption/synthetic_data).

Dataset "c_diss.csv" and "c_tot.csv" contained inside the folders [data_linear](diffusion_sorption/synthetic_data/data_linear), [data_freundlich](diffusion_sorption/synthetic_data/data_freundlich), and [data_langmuir](diffusion_sorption/synthetic_data/data_langmuir) are training datasets generated with the top Dirichlet boundary condition `solubility = 1.0`.

Dataset "c_diss_test.csv" and "c_tot_test.csv" contained inside the folders [data_linear](diffusion_sorption/synthetic_data/data_linear), [data_freundlich](diffusion_sorption/synthetic_data/data_freundlich), and [data_langmuir](diffusion_sorption/synthetic_data/data_langmuir) are test datasets generated with the top Dirichlet boundary condition `solubility = 1.0`.

## Experimental Dataset
All the codes and dataset can be found [here](diffusion-sorption/experimental_data).

Dataset "data_core1.xlsx" and "data_core2.xlsx" are experimental data obtained from core samples #1 and #2, respectively. Each file contains two sheets: "data" and "params". The sheet "data" contains the breakthrough curve data of the dissolved concentration, with the first column containing the time in days and the second column containing the dissolved concentration in kg/m^3. The sheet "params" contains the soil parameters and the simulation domain data.

Dataset "data_core2_long.xlsx" are experimental data obtained from core samples #2B. This file contains two sheets: "data" and "params". The sheet "data" contains the profile data of the total concentration, with the first column containing the depth in m and the second column containing the total concentration in kg/m^3. The sheet "params" contains the soil parameters and the simulation domain data.