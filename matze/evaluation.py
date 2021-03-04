"""
This script evaluates a desired number of models on a specified data set
(either the one the model has been trained on or an unseen one). Errors and
standard deviations over all models are printed to console and predictions are
stored in a .csv file
"""


import multiprocessing as mp
import numpy as np
import os
from utils import utils
import sys


# TCN
# ---- no boundary feed ---
# noise     seen                            unseen
# 0.0       0.12159859+-0.34860754          0.12404573+-0.24041595
# 1e-5      0.0061222245+-0.0041741757      0.047852196+-0.008689816
# ---- with boundary feed ----
# noise     seen                            unseen
# 0.0       0.09847083+-0.2801934			0.09194392+-0.20221253
# 1e-5      0.005890298+-0.0041398425       0.030304825+-0.011882976

# CONVLSTM
# ---- no boundary feed ----
# noise     seen                            unseen
# 0.0       0.100454226+-0.11223795         0.13087702+-0.11448048
# 1e-5      0.055559326+-0.06549839         0.08911292+-0.07485268
# ---- with boundary feed ----
# noise     seen                            unseen
# 0.0       0.06406576+-0.09415116			0.077789016+-0.09328022
# 1e-5      0.048843954+-0.05720509         0.06589377+-0.07942901

# DISTANA
# ---- no boundary feed ----
# noise     seen                            unseen
# 0.0       0.10598032+-0.31299484          0.10097959+-0.21704365
# 1e-5      0.05825447+-0.16827264          0.069869+-0.1109172
# ---- with boundary feed ----
# noise     seen                            unseen
# 0.0       0.09141965+-0.26532543			0.072504+-0.19800977
# 1e-5      0.009989682+-0.029078376        0.01635728+-0.04007094


#####################
# GLOBAL PARAMETERS #
#####################

# Model that is evaluated
MODEL_TYPE = "distana"  # Can be "convlstm", "distana" or "tcn"

# Testing parameters
NUMBER_OF_MODELS = 10
POOLSIZE = 1
SEQUENCE_LENGTH = 2000
DATA_NOISE = 1e-5
DATA_FILE = "seen"  # can be either "seen" or "unseen"
FEED_BOUNDARY_DATA = True
VISUALIZE_RESULTS = True
WRITE_DATA_TO_CSV = False

# Add the path of the chosen model to the system path to import the appropriate
# test and configuration files
sys.path.append("models/" + MODEL_TYPE)
import tester
import configuration as cfg

#############
# FUNCTIONS #
#############


def my_mse(a, b):
    return np.mean(np.square(a - b))


def write_to_csv(data_name, data):

    # Set up the path where the data are written to
    src_path = "data/results/" + str(cfg.ARCHITECTURE_NAME) + "/" \
               + DATA_FILE + "/train_noise" + str(DATA_NOISE)

    # Create directory if it does not yet exist
    os.makedirs(src_path, exist_ok=True)

    # Specify the pathname
    filepath = src_path + "/" + data_name + "_" + cfg.ARCHITECTURE_NAME

    if FEED_BOUNDARY_DATA:
    	# Add "boundary_feed" to filename
    	filepath = filepath + "_bound-feed"

    if "preds" in data_name:
        # Predictions of the n models
        for idx, sample in enumerate(data):
            np.savetxt(filepath + str(idx) + ".csv", sample, delimiter=",")

    else:
        # Labels
        np.savetxt(filepath + ".csv", data[0], delimiter=",")


def train_model(model_number):
	# This function is called by each worker of the multiprocessing pool to
	# perform a distinct model
	data = tester.run_testing(
		model_name=model_number,
		sequence_length=SEQUENCE_LENGTH,
        data_noise=DATA_NOISE,
        data_file=DATA_FILE,
        visualize_results=VISUALIZE_RESULTS,
        feed_boundary_data=FEED_BOUNDARY_DATA
	)
	return data


def main():

	#
    # Parallel evaluation

    # Set up a list holding the arguments for each training run
    arguments = []
    for model_number in range(NUMBER_OF_MODELS):
    	arguments.append(model_number)

    # Set up the multiprocessing pool
    with mp.Pool(POOLSIZE) as pool:

    	# Perform parallel training
    	results = pool.map(train_model, arguments)

    	# Wait for all processes to end
    	pool.terminate()
    	pool.join()

    # Convert the results list into an array
    results = np.array(results)

    # Separate the results array into dissolve and total concentration outputs
    # and targets
    outputs_dis, outputs_tot = results[:, 0, :, :, 0], results[:, 0, :, :, 1]
    targets_dis, targets_tot = results[:, 1, :, :, 0], results[:, 1, :, :, 1]

    # Write results to .csv documents
    if WRITE_DATA_TO_CSV:
        write_to_csv(data_name="diss_preds", data=outputs_dis)
        write_to_csv(data_name="tot_preds", data=outputs_tot)
        write_to_csv(data_name="diss_labels", data=targets_dis)
        write_to_csv(data_name="tot_labels", data=targets_tot)

    # Compute the mean squared error for each of the n models
    mse_list = [my_mse(outputs_dis[i], targets_dis[i])
                for i in range(NUMBER_OF_MODELS)]

    # Dump each model's mse score to console
    print(np.round(np.array(mse_list), 6))

    # Extract statistics from the mse_list
    mse_avg = np.mean(mse_list)
    mse_std = np.std(mse_list)

    # Dump final error and standard deviation to console
    print(str(mse_avg) + "+-" + str(mse_std))

    if VISUALIZE_RESULTS:
        utils.animate_diffusion(outputs_dis, outputs_tot, targets_dis,
                                targets_tot, cfg.TEACHER_FORCING_STEPS)


##########
# SCRIPT #
##########

# Execute main function when this script is called
if __name__ == "__main__":
    main()
