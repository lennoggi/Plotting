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
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_x[0]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_x[0]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_x[0]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_x[0]_mm08_large_14rl_NewCooling.asc"
)

filenames_y1 = (
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_y[0]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_y[0]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_y[0]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_y[0]_mm08_large_14rl_NewCooling.asc"
)

filenames_z1 = (
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_z[0]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_z[0]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_z[0]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_z[0]_mm08_large_14rl_NewCooling.asc"
)

filenames_x2 = (
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_x[1]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_x[1]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_x[1]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_x[1]_mm08_large_14rl_NewCooling.asc"
)

filenames_y2 = (
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_y[1]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_y[1]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_y[1]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_y[1]_mm08_large_14rl_NewCooling.asc"
)

filenames_z2 = (
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_NonSpinning_large_TaperedCooling_NewCooling/pt_loc_z[1]_NonSpinning_large_TaperedCooling_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pp08_large_14rl_NewCooling/pt_loc_z[1]_pp08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_pm08_large_14rl_NewCooling/pt_loc_z[1]_pm08_large_14rl_NewCooling.asc",
    "/home1/07825/lennoggi/Git_repositories/CactusAnalysis/Utils/data_mm08_large_14rl_NewCooling/pt_loc_z[1]_mm08_large_14rl_NewCooling.asc"
)


t_cols     = (8, 8, 8, 8)
coord_cols = (12, 12, 12, 12)

colors = ("dodgerblue", "indianred", "forestgreen", "magenta")

labels = (
    "Non-spinning", "++0.8", "+-0.8", "--0.8"
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


plt.figure() ##(figsize = (10., 4.))
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
    # XXX
    ##if n == 0: print(sep[np.where(t >= 13784.)])
    ##if n == 1: print(t[np.where(sep <= 9.23729585e+00)])
    # XXX

    plt.grid(linestyle = "--", linewidth = 0.5, alpha = 0.5)

    # XXX
    ##plt.plot(t, sep, linestyle = "-", linewidth = 1., marker = "", color = colors[n], label = labels[n])
    plt.plot(99189.9 + t, sep, linestyle = "-", linewidth = 1., marker = "", color = colors[n], label = labels[n])
    # XXX

plt.legend()
plt.tight_layout()
plt.savefig(figname)
plt.close()

print("Plot saved as '" + figname + "'")
