"""
This script take a particular bunch of .csv files containing the data of a
spedivied model (tcn, convlstm, distana, finn) and creates an animation
"""

import numpy as np
from utils import utils
import csv
import glob
import os
import shutil
import matplotlib.pyplot as plt



#####################
# GLOBAL PARAMETERS #
#####################

# Animate frames with matplotlib or write .png pictures to file and create .mp4
# with ffmpeg
VIDEO_TYPE = "ffmpeg"  # "matplotlib" or "ffmpeg"

# Model that is evaluated
MODEL_TYPE = "finn"  # Can bei "tcn", "convlstm", "distana" or "finn"

# Testing parameters
NUMBER_OF_MODELS = 10
DATA_NOISE = 1e-5
DATA_FILE = "unseen"  # can be either "seen" or "unseen"
FEED_BOUNDARY_DATA = False
TEACHER_FORCING_STEPS = 1
SEQUENCE_LENGTH = 700


#############
# FUNCTIONS #
#############


def read_csv_files(path_list):

    # Set up a list to store the data
    lst = []

    #
    # Read the required .csv files
    for path in path_list:

        # Initialize a list to store the rows
        data = []
        
        with open(path) as csv_file:
            
            # Ope the csv file and iterate over all its rows
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:

                # Convert the comma separated strings into a float list
                float_list = [float(x.split(",")[0]) for x in row]

                data.append(float_list[:SEQUENCE_LENGTH])
        
        # Convert the data with the rows into an array and append it to the
        # list
        lst.append(np.array(data))

    return np.array(lst)


def main():

    #
    # Determine the paths for the files
    path = "data/results/" + MODEL_TYPE + "/" + DATA_FILE + "/train_noise"\
           + str(DATA_NOISE) + "/"

    file_paths = np.sort(glob.glob(path + "*"))

    # Separate the paths into predictions for dissolved and total concentration
    if not FEED_BOUNDARY_DATA:
        dis_preds_paths = [path for path in file_paths
                           if "diss_preds" in path and "bound" not in path]
        tot_preds_paths = [path for path in file_paths
                           if "tot_preds" in path and "bound" not in path]
    else:
        dis_preds_paths = [path for path in file_paths
                           if "diss_preds" in path and "bound" in path]
        tot_preds_paths = [path for path in file_paths
                           if "tot_preds" in path and "bound" in path]

    # Set up the paths for the labels
    dis_label_paths = [path + "diss_labels_" + MODEL_TYPE + ".csv"]
    tot_label_paths = [path + "tot_labels_" + MODEL_TYPE + ".csv"]

    #
    # Read the required .csv files
    outputs_dis = read_csv_files(dis_preds_paths)
    outputs_tot = read_csv_files(tot_preds_paths)
    targets_dis = read_csv_files(dis_label_paths)
    targets_tot = read_csv_files(tot_label_paths)

    #
    # Create a video

    if VIDEO_TYPE == "matplotlib":
        # Animate the data
        utils.animate_diffusion(outputs_dis, outputs_tot, targets_dis,
                                targets_tot, TEACHER_FORCING_STEPS)
    elif VIDEO_TYPE == "ffmpeg":

        # Create temporary directory to store the pictures
        os.makedirs("tmp", exist_ok=True)

        # Iteratively write .png pictures to file
        for t in range(SEQUENCE_LENGTH):
            print("Creating picture " + str(t) + "/" + str(SEQUENCE_LENGTH))

            fig, outputs_dis_lines, outputs_tot_lines, targets_dis_line, \
              targets_tot_line, txt = utils.plot_diffusion(
                outputs_dis=outputs_dis,
                outputs_tot=outputs_tot,
                targets_dis=targets_dis,
                targets_tot=targets_tot,
                timestep=t,
                teacher_forcing_steps=TEACHER_FORCING_STEPS
            )
            plt.tight_layout()
            plt.savefig("tmp/" + str(t) + str(".png"), bbox_inches="tight",
                        pad_inches=0.1)
            plt.close()

        # Call ffmpeg to create a video from the separate pictures
        vid_name = MODEL_TYPE + "_" + DATA_FILE
        if FEED_BOUNDARY_DATA:
            vid_name = vid_name + "_bound-feed"

        os.system(
            "ffmpeg -r 35 -i tmp/%d.png " \
            "-vf pad=width=1210:height=610:x=10:y=7:color=black " \
            "-c:v libx264 -pix_fmt yuv420p " + vid_name + ".mp4"
        )

        # Remove the temporary directory (and its content)
        shutil.rmtree("tmp")


##########
# SCRIPT #
##########

# Execute main function when this script is called
if __name__ == "__main__":
    main()
