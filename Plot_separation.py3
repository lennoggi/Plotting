# =========================================================================
# This script plots the coordinate separation between two objects (e.g. two
# punctures)
# TODO: plot the proper separation as well
#       Problem: get the 3-metric components at the objects' locations
# =========================================================================

import numpy as np
from matplotlib import pyplot as plt


######################### USER-DEFINED PARAMETERS ##############################

filename_x1 = "/home1/07825/lennoggi/pt_loc_x[0].asc"
filename_y1 = "/home1/07825/lennoggi/pt_loc_y[0].asc"
filename_z1 = "/home1/07825/lennoggi/pt_loc_z[0].asc"

filename_x2 = "/home1/07825/lennoggi/pt_loc_x[1].asc"
filename_y2 = "/home1/07825/lennoggi/pt_loc_y[1].asc"
filename_z2 = "/home1/07825/lennoggi/pt_loc_z[1].asc"

t_col     = 0
coord_col = 1

plot_title = "++0.8 45-deg run"
plot_dir   = "/home1/07825/lennoggi"

################################################################################


data_x1 = np.loadtxt(filename_x1)
data_y1 = np.loadtxt(filename_y1)
data_z1 = np.loadtxt(filename_z1)

data_x2 = np.loadtxt(filename_x2)
data_y2 = np.loadtxt(filename_y2)
data_z2 = np.loadtxt(filename_z2)

l = len(data_x1)

assert(len(data_y1) == l)
assert(len(data_z1) == l)
assert(len(data_x2) == l)
assert(len(data_y2) == l)
assert(len(data_z2) == l)


t = data_x1[:, t_col]

x1 = data_x1[:, coord_col]
y1 = data_y1[:, coord_col]
z1 = data_z1[:, coord_col]
x2 = data_x2[:, coord_col]
y2 = data_y2[:, coord_col]
z2 = data_z2[:, coord_col]

sep_x = x2 - x1
sep_y = y2 - y1
sep_z = z2 - z1

sep = np.sqrt(sep_x*sep_x + sep_y*sep_y + sep_z*sep_z)


plot_fullpath = plot_dir + "/Coordinate_separation.pdf"

plt.figure()
plt.title(plot_title, fontsize = 15., fontweight = "bold",
          fontstyle = "normal", fontname = "Ubuntu", color = "midnightblue")
plt.xlabel("$t\,\left[M\\right]$")
##plt.xlabel("$t\,\left[ms\\right]$")
plt.ylabel("Coordinate separation $\left[M\\right]$")
plt.plot(t, sep,
##plt.plot(t*4.9257949707731345e-03, sep,
         linestyle = "-", marker = ".", markersize = 3., color = "dodgerblue")
plt.tight_layout()
plt.savefig(plot_fullpath)
plt.close()

print("Plot saved as '" + plot_fullpath + "'")
