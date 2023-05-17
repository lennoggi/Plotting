import numpy as np
from os import path
import argparse

# Parser setup
description = "Merge Cactus text files from different restarts of a given simulation"
parser      = argparse.ArgumentParser(description = description)

parser.add_argument("--simdir", required = True,
                    help = "Simulation directory containing all the 'output-xxxx' subdirectories")

parser.add_argument("--subpath", required = True,
                    help = "Path of the files to be merged relative to 'output-xxxx' (must be the same for all restarts)")

parser.add_argument("--outfile", required = True,
                    help = "Full path to the file storing the full dataset")

args    = parser.parse_args()
simdir  = args.simdir
subpath = args.subpath
outfile = args.outfile


# Loop over all restarts
fullpath = simdir + "/output-0000/" + subpath
n        = 0
fulldata = np.array([])

while (path.exists(fullpath)):
    assert(n >= 0)
    data = np.loadtxt(fullpath)

    if (n == 0):  fulldata = data
    else:         np.concatenate((fulldata, data), axis = 0)

    out_str = "output-" + str("{:0>4d}".format(n))
    print(out_str + " merged")

    fullpath = simdir + "/" + out_str + "/" + subpath
    n += 1


# Write the full dataset to file
np.savetxt(outfile, fulldata)
