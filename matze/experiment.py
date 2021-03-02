"""
This script trains a desired number of models in parallel and prints the final
average error and standard deviation over all models to console
"""

import multiprocessing as mp
import numpy as np
import sys


# CONVLSTM
# 10 models
# noise0.0  -> 5.655764040511713e-06+-9.597546675193882e-07
# noise1e-5 ->

# DISTANA
# 10 models
# noise0.0  -> 1.3350772432545455e-06+-8.511844531846674e-07
# noise1e-5 -> 1.8462115297950275e-06+-1.11033431494542e-06

# TCN
# 10 models
# noise0.0  -> 5.631153540434752e-06+-4.638821505382622e-06
# noise1e-5 ->


#####################
# GLOBAL PARAMETERS #
#####################

# Model that is trained
MODEL_TYPE = "tcn"  # Can bei "convlstm", "distana" or "tcn"

# Training parameters
DATA_NOISE = 1e-5
NUMBER_OF_MODELS = 10
POOLSIZE = 10

# Other parameters
PRINT_TRAINING_PROGRESS = True
SEQUENCE_LENGTH = 500

# Add the path of the chosen model to the system path to import the appropriate
# test and configuration files
sys.path.append("models/" + MODEL_TYPE)
import trainer
import configuration as cfg

#############
# FUNCTIONS #
#############

def train_model(model_number):
	# This function is called by each worker of the multiprocessing pool to
	# perform a distinct model
	epoch_errors = trainer.run_training(
		model_name=model_number,
		print_progress=PRINT_TRAINING_PROGRESS,
		sequence_length=SEQUENCE_LENGTH,
        data_noise=DATA_NOISE
	)
	return epoch_errors


def main():

	#
    # Parallel training

    # Set up a list holding the arguments for each training run
    arguments = []
    for model_number in range(NUMBER_OF_MODELS):
    	arguments.append(model_number)

    # Set up the multiprocessing pool
    with mp.Pool(POOLSIZE) as pool:

    	# Perform parallel training
    	error_lists = pool.map(train_model, arguments)

    	# Wait for all processes to end
    	pool.terminate()
    	pool.join()

    # Convert the error_lists into an array
    error_ary = np.array(error_lists)

    # Calculate training statistics
    mse_avg = np.mean(error_ary[:, -1])
    mse_std = np.std(error_ary[:, -1])

    # Dump final error and standard deviation to console
    print(str(mse_avg) + "+-" + str(mse_std))


##########
# SCRIPT #
##########

# Execute main function when this script is called
if __name__ == "__main__":
    main()
