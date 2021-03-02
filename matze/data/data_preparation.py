
import numpy as np
import os

c_diss = np.loadtxt("raw_txt/c_diss_test.txt")
c_tot = np.loadtxt("raw_txt/c_tot_test.txt")

c_diss = np.expand_dims(np.swapaxes(c_diss, axis1=0, axis2=1), axis=2)
c_tot = np.expand_dims(np.swapaxes(c_tot, axis1=0, axis2=1), axis=2)

data = np.concatenate((c_diss, c_tot), axis=2)

os.makedirs("numpy", exist_ok=True)
np.save("numpy/data_test.npy", data)
