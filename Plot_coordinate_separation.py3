# ============================================================================
# This script plots the coordinate separation between two objects (e.g. two
# punctures)
# NOTE: typically, you'll need the object's location data from all restarts of
#       the simulation, and you can use Plot_ASCII_merge.py3 for that
# ============================================================================

import numpy as np
from matplotlib import pyplot as plt


######################### USER-DEFINED PARAMETERS ##############################

filenames_x1 = (
    "pt_loc_x[0].asc"
##    "pt_loc_x[0]_old.asc"
)

filenames_y1 = (
    "pt_loc_y[0].asc"
##    "pt_loc_y[0]_old.asc"
)

filenames_z1 = (
    "pt_loc_z[0].asc"
##    "pt_loc_z[0]_old.asc"
)


filenames_x2 = (
    "pt_loc_x[1].asc"
##    "pt_loc_x[1]_old.asc"
)

filenames_y2 = (
    "pt_loc_y[1].asc"
##    "pt_loc_y[1]_old.asc"
)

filenames_z2 = (
    "pt_loc_z[1].asc"
##    "pt_loc_z[1]_old.asc"
)


t_cols     = (8, 8)
coord_cols = (12, 12)

colors = ("dodgerblue") ##, "red")

labels = (
    "New" ##, "Old"
)

figname = "Coordinate_separation.pdf"

################################################################################


N = len(filenames_x1)
assert N > 0

assert len(filenames_y1) == N
assert len(filenames_z1) == N
assert len(filenames_x2) == N
assert len(filenames_y2) == N
assert len(filenames_z2) == N

assert len(colors) == N
assert len(labels) == N


plt.figure()
plt.xlabel("$t\,\left[M\\right]$", fontsize = 12.)
plt.ylabel("$a\,\left[M\\right]$", fontsize = 12.)

for n in range(N):
    data_x1 = np.loadtxt(filenames_x1[n])
    data_y1 = np.loadtxt(filenames_y1[n])
    data_z1 = np.loadtxt(filenames_z1[n])

    data_x2 = np.loadtxt(filenames_x2[n])
    data_y2 = np.loadtxt(filenames_y2[n])
    data_z2 = np.loadtxt(filenames_z2[n])

    L = len(data_x1)

    assert(len(data_y1) == L)
    assert(len(data_z1) == L)
    assert(len(data_x2) == L)
    assert(len(data_y2) == L)
    assert(len(data_z2) == L)

    t = data_x1[:, t_cols[n]]
    coord_col = coord_cols[n]

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

    plt.grid(linestyle = "--", linewidth = 0.5, alpha = 0.5)

    plt.plot(t, sep, linestyle = "-", linewidth = 1., marker = "", color = colors[n], label = labels[n])

##plt.legend()
plt.tight_layout()
plt.savefig(figname)
plt.close()

print("Plot saved as '" + figname + "'")
