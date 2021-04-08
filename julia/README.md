# Finite Volume Neural Network
This is the code and data repository for the Finite Volume Neural Network method using the Julia language.

## Organization
This repository contains the implementation of the FINN method for the non-linear diffusion-sorption process.
The implementation is divided in two parts: synthetic and experimental data.

## Synthetic Dataset
All the codes and dataset can be found [here](diffusion_sorption/synthetic_data).

To generate new dataset, use [this code](diffusion_sorption/synthetic_data/data_generation.jl), and adjust the soil parameters, simulation domain, and working directory as necessary.

Dataset "c_diss.txt" and "c_tot.txt" contained inside the folders [data_linear](diffusion_sorption/synthetic_data/data_linear), [data_freundlich](diffusion_sorption/synthetic_data/data_freundlich), and [data_langmuir](diffusion_sorption/synthetic_data/data_langmuir) are training datasets generated with the same parameters as contained in the [data generation code](diffusion_sorption/synthetic_data/data_generation.jl), with the parameter "solubility = 1.0".

Dataset "c_diss_test.txt" and "c_tot_test.txt" contained inside the folders [data_linear](diffusion_sorption/synthetic_data/data_linear), [data_freundlich](diffusion_sorption/synthetic_data/data_freundlich), and [data_langmuir](diffusion_sorption/synthetic_data/data_langmuir) are test datasets generated with the same parameters as contained in the [data generation code](diffusion_sorption/synthetic_data/data_generation.jl), with the parameter "solubility = 0.7".

The code [hom_diff_sorp_sim.jl](diffusion_sorption/synthetic_data/hom_diff_sorp_sim.jl) is used to train the network with the full field solution (used for benchmarking purposes with other methods), and the code [hom_diff_sorp_sparse.jl](diffusion_sorption/synthetic_data/hom_diff_sorp_sparse.jl) is used to train the network with sparse dataset, which are the breakthrough curve of the dissolved concentration, and the total concentration profile at the last time step.

The code [network_test.jl](diffusion_sorption/synthetic_data/network_test.jl) is used to test the trained network. Adjust the save_folder to where the saved parameter file is located.

## Experimental Dataset
All the codes and dataset can be found [here](diffusion_sorption/experimental_data).

Dataset "data_core1.xlsx" and "data_core2.xlsx" are experimental data obtained from core samples #1 and #2, respectively. Each file contains two sheets: "data" and "params". The sheet "data" contains the breakthrough curve data of the dissolved concentration, with the first column containing the time in days and the second column containing the dissolved concentration in kg/m^3. The sheet "params" contains the soil parameters and the simulation domain data.

Dataset "data_core2_long.xlsx" are experimental data obtained from core samples #2B. This file contains two sheets: "data" and "params". The sheet "data" contains the profile data of the total concentration, with the first column containing the depth in m and the second column containing the total concentration in kg/m^3. The sheet "params" contains the soil parameters and the simulation domain data.

The code [experimental_data.jl](diffusion_sorption/experimental_data/experimental_data.jl) is used to train the network with the breakthrough curve of core sample #2.

The code [experiment_test_1.jl](diffusion_sorption/experimental_data/experiment_test_1.jl) and [experiment_test_2B.jl](diffusion_sorption/experimental_data/experiment_test_2B.jl) are used to test the trained network with data from core samples #1 and #2B, respectively. Adjust the save_folder to where the saved parameter file is located.
