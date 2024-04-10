import numpy as np
from matplotlib import pyplot as plt
from postcactus.simdir import SimDir
from postcactus import grid_data as gd
import warnings
import argparse


# ============
# Parser setup
# ============
description = "Plot 1D HDF5 Cactus data"
parser      = argparse.ArgumentParser(description = description)

parser.add_argument("--simdirs", nargs = "+", required = True,
                    help = "Full path(s) to the simulation directories containing the data to be plotted")
parser.add_argument("--labels", nargs = "+", required = True,
                    help = "Plot labels for each simulation directory")
parser.add_argument("--gf", required = True,
                    help = "Cactus grid function to be plotted from each simulation directory")
parser.add_argument("--gfsymb", required = True,
                    help = "Symbol for the Cactus grid function to be plotted from each simulation directory (used for the label on the y axis)")
parser.add_argument("--iteration", type = int, required = True,
                    help = "Iteration for which the data will be plotted")
parser.add_argument("--direction", choices = ["x", "y", "z"], required = True,
                    help = "Direction for which the data will be plotted")
parser.add_argument("--units", choices = ["arbitrary", "geometric", "SI"], required = True,
                    help = "Units used for the plots")

args      = parser.parse_args()
simdirs   = args.simdirs
labels    = args.labels
gf        = args.gf
gfsymb    = args.gfsymb
iteration = args.iteration
direction = args.direction
units     = args.units

nsims = len(simdirs)
assert(len(labels) == nsims)


# ============
# Units setup
# ============
if units == "arbitrary":
    conv_fac_time  = 1.
    unit_time_str  = "$\mathbf{M}$"

    conv_fac_space = 1.
    unit_space_str = "[$\mathbf{M}$]"

    # TODO: add other potentially useful conversion factors
    if gf == "rho" or gf == "rho_b":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M}^{-2}\\right]$"
    elif gf == "press" or gf == "P":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M}^{2}\\right]$"
    elif gf == "eps":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M}\\right]$"
    elif gf == "smallb2" or gf == "b2small" or gf == "B_norm":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M}^{-1}\\right]$"
    else:
        conv_fac_gf = 1.
        unit_gf_str = ""
        warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'")


elif units == "geometric":
    conv_fac_time  = 1.
    unit_time_str  = "$\mathbf{M_{\\odot}}$"

    conv_fac_space = 1.
    unit_space_str = "[$\mathbf{M_{\\odot}}$]"

    # TODO: add other potentially useful conversion factors
    if gf == "rho" or gf == "rho_b":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{-2}\\right]$"
    elif gf == "press" or gf == "P":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{2}\\right]$"
    elif gf == "eps":
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}\\right]$"
    else:
        conv_fac_gf = 1.
        unit_gf_str = ""
        warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'")


elif units == "SI":
    G    = 6.67408e-11;     # m^3/(kg·s^2)
    c    = 2.99792458e+08;  # m/s
    Msun = 1.98847e+30;     # kg

    GMsun_over_c2 = G*Msun/(c*c)

    Msun_to_m          = GMsun_over_c2                  # Length
    Msun_to_s          = GMsun_over_c2/c                # Time
    Msun_to_kg_over_m3 = (GMsun_over_c2**3)/Msun        # Mass density
    Msun_to_N_over_m2  = Msun_to_kg_over_m3/(c*c)       # Pressure
    Msun_to_N_over_A2  = np.sqrt(Msun_to_N_over_m2/mu0) # Magnetic field

    conv_fac_space = 0.001*Msun_to_m  # From solar masses to kilometers
    conv_fac_time  = 1000.*Msun_to_s  # From solar masses to milliseconds

    unit_time_str  = " $\mathbf{ms}$"
    unit_space_str = "[$\mathbf{km}$]"

    # TODO: add other potentially useful conversion factors
    if gf == "rho" or gf == "rho_b":
        conv_fac_gf = Msun_to_kg_over_m3
        unit_gf_str = "$\,\left[\\frac{kg}{m^3}\\right]$"
    elif gf == "press" or gf == "P":
        conv_fac_gf = Msun_to_N_over_m2
        unit_gf_str = "$\,\left[\\frac{N}{m^2}\\right]$"
    elif gf == "smallb2" or gf == "b2small" or gf == "B_norm":
        conv_fac_gf = Msun_to_N_over_A2
        unit_gf_str = "$\,\left[T\\right]$"
    else:
        conv_fac_gf = 1.
        unit_gf_str = ""
        warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'")


else: raise RuntimeError("Unrecognized units '" + units + "'. This should have been caught by the parser.")
 


# ====
# Plot
# ====
plt.figure()
##plt.title("")
plt.xlabel(direction + "$\,$" + unit_space_str, fontsize = 12.)
plt.ylabel(gfsymb    + "$\,$" + unit_gf_str,    fontsize = 12.)

for n in range(nsims):
    simdir = simdirs[n]
    sd     = SimDir(simdir)

    if   direction == "x": data = sd.grid.x
    elif direction == "y": data = sd.grid.y
    elif direction == "z": data = sd.grid.z
    else: raise RuntimeError("Invalid direction '" + direction + "' specified. This should have been caught by the parser")

    print("Available data along direction " + direction + " inside directory '" + simdir + "':")
    print(data)

    gfdata        = data.read(gf, iteration)
    coord, gfdata = gd.merge_comp_data_1d(gfdata)

    plt.plot(coord*conv_fac_space, gfdata*conv_fac_gf, marker = ".", markersize = 3.,
             linestyle = "-", linewidth = 1., color = "dodgerblue", label = labels[n])

plt.legend()
plt.tight_layout()
plt.savefig(gf + ".pdf")
plt.close()
