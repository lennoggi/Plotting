################################################################################
#
#  ***** VERY IMPORTANT *****
#  --------------------------
#  In order to prevent PostCactus from aborting as soon as it finds a dataset
#  with no more iterations available, edit the PostCactus file
#
#      ~/.local/lib/python3.7/site-packages/postcactus/cactus_grid_h5.py
#
# in the following way:
#   1. add the statement
#          import warnings
#      at the beginning
#
#   2. replace line 686, i.e.
#          raise ValueError("Iteration %d not in available range" % it)
#      with
#          warnings.warn("Iteration %d not in available range" % it)
#
#   3. replace line 906, i.e.
#          raise RuntimeError("Could not read iteration %d for %s" % (it, name))
#      with
#          warnings.warn("Could not read iteration %d for %s" % (it, name))
#
################################################################################


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib.patches import Rectangle
from postcactus.simdir import SimDir
from postcactus import grid_data as gd
from postcactus import visualize as viz
from scipy.spatial import ConvexHull
from os import path
import warnings

from matplotlib import rc
from matplotlib import rcParams
#rc("mathtext", usetex = True)
#rc("font",**{"family":"sans-serif","sans-serif":["Avant Garde"]})
#rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
rcParams["mathtext.fontset"] = "dejavuserif"





######################### USER-DEFINED PARAMETERS ##############################

# **********************
# *  General settings  *
# **********************

# Directories containing the files to be opened
# ---------------------------------------------
data_dirs = np.array([
    ##"/lagoon/bbhdisk/CBD_SphericalNR/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP"
    "/lagoon/bbhdisk/BBHDiskMerger/CBD_handoff_IGM_McLachlan_Spinning_aligned08"
])


# Directory where the plots will be placed
# ----------------------------------------
##plots_dir = "/lagoon/lennoggi/Snapshots/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP_ZoomOut"
plots_dir = "/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_TEST_PLOT_GRID"


# Which grid functions to plot
# ----------------------------
grid_functions = np.array([
    "rho_b"  ##"rho"
])


# Input coordinates
# -----------------
input_coords = "Cartesian"  # "Cartesian" or "Exponential fisheye"


# Which 2D slices to plot
# Cartesian      coordinates: xy, xz or yz plane
# Spherical-like coordinates: r-theta, r-phi or theta-phi plane
# -------------------------------------------------------------
planes = np.array([
    "xy"  ##"xz"
])


# Plot absolute values?
# ---------------------
abs_vals = np.array([
    False
])


# Iterations and initial time info
# --------------------------------
first_it    = 187392  ##151552  ##115712
last_it     = 191488 #222208  ##186368  ##150528  ##1000000000  # Set this to a huge number to plot all iterations
out2D_every = 1024    ##400
t0          = 119977.65  ##0.

# Do you want to find the max and min in the data for every available
# iteration?
# NOTE: this may take some time
# -------------------------------------------------------------------
compute_min_max = np.array([
    False
])





# ********************************
# *  Figure and layout settings  *
# ********************************

# Plot extent **IN NUMERICAL COORDINATES**; provide it as
# [xmin, xmax, ymin, ymax]. Used to build the PostCactus geometry.
# ----------------------------------------------------------------
plot_extents = np.array([
     ##np.array([np.log(15.1), np.log(2000.), 0., 2.*np.pi])
     np.array([-40., 40., -40., 40.])
])


# Actual plot extent if the input coordinates are not Cartesian, i.e., what you
# will actually see in the snapshot(s)
# NOTE: set to None if you want to keep the original grid dimensions when using
#       non-Cartesian coordinate systems
# NOTE: used as the starting plot extent if 'zoom' (see below) is True
actual_plot_extents = np.array([
    np.array([-40., 40., -40., 40.])
##    np.array([-300., 300., -300., 300.])
##    np.array([-2010., 2010., -2010., 2010.])
])


# Subplots layout
# ---------------
nsubplots_x = 1
nsubplots_y = 1

# Figure size and resolution
# --------------------------
figsize = [12., 10.]
dpi     = 200

# File extension for the plots
# ----------------------------
fig_ext = ".png"





# *******************************
# *  Apparent horizon settings  *
# *******************************

# Draw the apparent horizon(s)?
# -----------------------------
draw_AH = np.array([
    True  ##False
])


# How many files per AH are there, i.e. the maximum value of 'AH_number' in
# 'h.t<iteration>.ah<AH_number>'
# -------------------------------------------------------------------------
N_AH_files = 2


# **IMPORTANT**
# Make sure all AH files ('h.t<iteration>.ah<AH_number>') are in the same
# directory. In case they live under different 'output-xxxx' directories, copy
# them to a common directory.
# ----------------------------------------------------------------------------
AH_dirs = np.array([
    ##"/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/AH_data",
    "/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08/AH_data"
])





# ********************
# *  Scale settings  *
# ********************

# Use a logarithmic scale?
# ------------------------
logscale = np.array([
    True
])


# Use a symmetric logarithmic scale? I.e., if a logarithmic scale is in use and
# data values extend down to zero, then a linear scale is used from zero to the
# desired minimum in the colorbar (see below)
# -----------------------------------------------------------------------------
symlogscale = np.array([
    False
])


# If a linear color scale is in use (i.e. if 'logscale' is false), normalize it
# between 0 and 1?
# -----------------------------------------------------------------------------
linscale_norm = np.array([
    True
])





# *****************
# *  Plot setup  *
# *****************

# Units
# -----
# 1 solar mass time     = 4.9257949707731345e-03 ms
# 1 solar mass distance = 1.4767161818921162 km
units = "arbitrary"  # "arbitrary", "geometric" or "SI"


# Names of the variables to be put close to the colorbar
# ------------------------------------------------------
varnames = np.array([
    "$\\rho$"
])


# Titles for each subplot
# -----------------------
titles = np.array([
    ""
])


# Add colorbar(s)?
# ----------------
add_clb = np.array([
    True
])


# Extent of the color scales (note that the actual scale may extend below
# colorbar_extents[i][0] if logscale[i] = "yes" and symlogscale[i] 0 "yes")
# -------------------------------------------------------------------------
clb_extents = np.array([
    np.array([1.e-08, 1.5e-02])
])


# Type of colorbar extension outside its limits ("neither", "max", "min" or
# "both")
# -------------------------------------------------------------------------
clb_extends = np.array([
    "max"
])


# Colormap
# --------
cmaps = np.array([
    "plasma"
])


# Title options
# -------------
titlecolor       = "midnightblue"
titlepad         = 1.02
title_fontsize   = 35.
title_fontweight = "bold"
title_fontstyle  = "normal"
title_fontname   = "Ubuntu"

# Labels options
# --------------
labelsize  = 25.
labelpad_x = 3.
labelpad_y = -5.
ticksize   = 20.

# Colorbar label options
# ----------------------
clb_ticksize        = 20.
clblabel_pad        = 8.
clb_fraction        = 0.05
clblabel_fontsize   = 25.
clblabel_fontweight = "bold"
clblabel_fontstyle  = "normal"

# Iteration and time strings options
# ----------------------------------
it_pos             = np.array([0.15, 0.015])
time_pos           = np.array([0.6, 0.015])
it_time_fontsize   = 25.
it_time_fontweight = "bold"
it_time_fontstyle  = "normal"





# *****************************
# *  Dynamic zooming options  *
# *****************************

# Zoom in/out as time goes?
# -------------------------
zooms = np.array([
    False  ##True
])


# If zooming in/out, set the actual plot extents at the final time here
# ---------------------------------------------------------------------
actual_plot_extents_end = np.array([
    ##np.array([-2010., 2010., -2010., 2010.])
    ##np.array([-300., 300., -300., 300.])
    np.array([-40., 40., -40., 40.])
])


# Iterations at which zooming in/out should begin/end
# -------------------------------------------------------
first_its_zoom = np.array([
    187392 ##535200
])

last_its_zoom = np.array([
    187393 ##633600
])





# ************************************************
# *  Dynamic grid plotting (pcolormesh) options  *
# ************************************************

# Plot the grid using the 'edgecolor' option in np.pcolormesh?
# NOTE: this works best on uniform grids, as it shows the full
#       mesh and not just the refinement level boundaries
# ------------------------------------------------------------
plot_grid = np.array([
    False  ##True
])


# If plot_grid is true, do you want the grid to gradually fade in/out?
# --------------------------------------------------------------------
vary_grid_transparency = np.array([
    True
])


# Iterations at which the change in grid transparency should begin/end
# --------------------------------------------------------------------
first_its_alpha_grid = np.array([
    187392  ##736000
])

last_its_alpha_grid = np.array([
    187393  ##769600
])


# Grid transparency values at the beginning/end
# ---------------------------------------------
alpha_grid_init = np.array([
    0.
])

alpha_grid_end = np.array([
    1.
])





# ***********************************************************************
# *  Dynamic refinement level boundaries plotting (PostCactus) options  *
# ***********************************************************************

# Plot refinement level boundaries?
# ---------------------------------
plot_reflevels = np.array([
    True
])


# If plot_reflevels is true, do you want the refinement level boundaries to
# gradually fade in/out?
# -------------------------------------------------------------------------
vary_reflevels_transparency = np.array([
    True
])


# Iterations at which the change in refinement levels transparency should
# begin/end
# -----------------------------------------------------------------------
first_its_alpha_reflevels = np.array([
    187392
])

last_its_alpha_reflevels = np.array([
    191488 #222208
])


# Refinement levels transparency at the beginning/end
# ---------------------------------------------------
alpha_reflevels_init = np.array([
    1.  ##0.
])

alpha_reflevels_end = np.array([
    0.
])


# Range of refinement levels boundaries to be plotted
# NOTE: the minimum must be no smaller than 0, and you can set the maximum to
#       some high value to plot all reflevel boundaries
# ---------------------------------------------------------------------------
reflevel_ranges = np.array([
    np.array([0, 20])
])

################################################################################










######################### CODE SELF-PROTECTION #################################

N_datasets = len(data_dirs)

assert(len(grid_functions) == N_datasets)

assert(input_coords == "Cartesian" or
       input_coords == "Exponential fisheye")

assert(len(planes) == N_datasets)
for plane in planes:
    assert(plane == "xy" or plane == "xz" or plane == "yz")

assert(len(abs_vals) == N_datasets)
for abs_val in abs_vals:
    assert(abs_val or not abs_val)

assert(first_it    >= 0)
assert(last_it     >= first_it)
assert(out2D_every >= 0)

assert(len(compute_min_max) == N_datasets)
for comp in compute_min_max:
    assert(comp or not comp)

assert(len(plot_extents) == N_datasets)
for plot_extent in plot_extents:
    assert(len(plot_extent == 4))
    assert(plot_extent[1] > plot_extent[0])
    assert(plot_extent[3] > plot_extent[2])

assert(len(actual_plot_extents) == N_datasets)
for actual_plot_extent in actual_plot_extents:
    assert(len(actual_plot_extent == 4))
    assert(actual_plot_extent[1] > actual_plot_extent[0])
    assert(actual_plot_extent[3] > actual_plot_extent[2])

assert(nsubplots_x > 0)
assert(nsubplots_y > 0)
assert(nsubplots_x*nsubplots_y == N_datasets)

assert(len(figsize) == 2)
assert(dpi > 0)

assert(len(draw_AH) == N_datasets)
for draw in draw_AH:
    assert(draw or not draw)

assert(N_AH_files > 0)
assert(len(AH_dirs) == N_datasets)

assert(len(logscale) == N_datasets)
for log in logscale:
    assert(log or not log)

assert(len(symlogscale) == N_datasets)
for symlog in symlogscale:
    assert(symlog or not symlog)

assert(len(linscale_norm) == N_datasets)
for lin in linscale_norm:
    assert(lin or not lin)

assert(units == "arbitrary" or
       units == "geometric" or
       units == "SI")

assert(len(varnames) == N_datasets)
assert(len(titles) == N_datasets)

assert(len(add_clb) == N_datasets)
for aclb in add_clb:
    assert(aclb or not aclb)

assert(len(clb_extents) == N_datasets)
for clb_extt in clb_extents:
    assert(len(clb_extt) == 2)
    assert(clb_extt[1] > clb_extt[0])

assert(len(clb_extends) == N_datasets)
for clb_extd in clb_extends:
    assert(clb_extd == "min"  or
           clb_extd == "max"  or
           clb_extd == "both")

assert(len(cmaps) == N_datasets)

assert(title_fontsize    > 0.)
assert(labelsize         > 0.)
assert(ticksize          > 0.)
assert(clb_ticksize      > 0.)
assert(clb_fraction      > 0.)
assert(clblabel_fontsize > 0.)

assert(len(it_pos   == 2))
assert(len(time_pos == 2))
assert(it_time_fontsize > 0.)

assert(len(zooms) == N_datasets)
for zm in zooms:
    assert(zm or not zm)

assert(len(actual_plot_extents_end) == N_datasets)
for actual_plot_extent_end in actual_plot_extents_end:
    assert(len(actual_plot_extent_end == 4))
    assert(actual_plot_extent_end[1] > actual_plot_extent_end[0])
    assert(actual_plot_extent_end[3] > actual_plot_extent_end[2])

assert(len(first_its_zoom) == N_datasets)
for first_it_zoom in first_its_zoom:
    assert(first_it_zoom >= first_it)

assert(len(last_its_zoom) == N_datasets)
for last_it_zoom in last_its_zoom:
    assert(last_it_zoom <= last_it)

assert(len(plot_grid) == N_datasets)
for pg in plot_grid:
    assert(pg or not pg)

assert(len(vary_grid_transparency) == N_datasets)
for vgt in vary_grid_transparency:
    assert(vgt or not vgt)

assert(len(first_its_alpha_grid) == N_datasets)
for first_it_alpha_grid in first_its_alpha_grid:
    assert(first_it_alpha_grid >= first_it)

assert(len(last_its_alpha_grid) == N_datasets)
for last_it_alpha_grid in last_its_alpha_grid:
    assert(last_it_alpha_grid <= last_it)

assert(len(alpha_grid_init) == N_datasets)
for agi in alpha_grid_init:
    assert(agi >= 0.)

assert(len(alpha_grid_end) == N_datasets)
for age in alpha_grid_end:
    assert(age >= 0.)

assert(len(plot_reflevels) == N_datasets)
for plot_rlvl in plot_reflevels:
    assert(plot_rlvl or not plot_rlvl)

assert(len(vary_reflevels_transparency) == N_datasets)
for vrt in vary_reflevels_transparency:
    assert(vrt or not vrt)

assert(len(first_its_alpha_reflevels) == N_datasets)
for first_it_alpha_reflevels in first_its_alpha_reflevels:
    assert(first_it_alpha_reflevels >= first_it)

assert(len(last_its_alpha_reflevels) == N_datasets)
for last_it_alpha_reflevels in last_its_alpha_reflevels:
    assert(last_it_alpha_reflevels <= last_it)

assert(len(alpha_reflevels_init) == N_datasets)
for ari in alpha_reflevels_init:
    assert(ari >= 0.)

assert(len(alpha_reflevels_end) == N_datasets)
for are in alpha_reflevels_end:
    assert(are >= 0.)

assert(len(reflevel_ranges) == N_datasets)
for rr in reflevel_ranges:
    assert(len(rr) == 2)
    assert(rr[1] >= rr[0])

################################################################################










############################## UNITS SETUP #####################################

if (units == "arbitrary"):
    conv_fac_time  = 1.
    unit_time_str  = "$\mathbf{M}$"

    conv_fac_space = 1.
    unit_space_str = "[$\mathbf{M}$]"

    # TODO: add magnetic field and any other useful conversions
    for gf in grid_functions:
        if (gf == "rho"   or gf == "rho_b"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{-2}\\right]$"
        elif (gf == "press" or gf == "P"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{2}\\right]$"
        elif (gf == "eps"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}\\right]$"
        elif (gf == "smallb2" or gf == "b2small" or gf == "B_norm"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{-1}\\right]$"
        else:
            conv_fac_gf = 1.
            unit_gf_str = ""
            warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'") 


elif (units == "geometric"):
    conv_fac_time  = 1.
    unit_time_str  = "$\mathbf{M_{\\odot}}$"

    conv_fac_space = 1.
    unit_space_str = "[$\mathbf{M_{\\odot}}$]"

    # TODO: add magnetic field and any other useful conversions
    for gf in grid_functions:
        if (gf == "rho"   or gf == "rho_b"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{-2}\\right]$"
        elif (gf == "press" or gf == "P"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{2}\\right]$"
        elif (gf == "eps"):
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}\\right]$"
        else:
            conv_fac_gf = 1.
            unit_gf_str = ""
            warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'") 


elif (units == "SI"):
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

    # TODO: add magnetic field and any other useful conversion
    for gf in grid_functions:
        if (gf == "rho"   or gf == "rho_b"):
            conv_fac_gf = Msun_to_kg_over_m3
            unit_gf_str = "$\,\left[\\frac{kg}{m^3}\\right]$"
        elif (gf == "press" or gf == "P"):
            conv_fac_gf = Msun_to_N_over_m2
            unit_gf_str = "$\,\left[\\frac{N}{m^2}\\right]$"
        elif (gf == "smallb2" or gf == "b2small" or gf == "B_norm"):
            conv_fac_gf = Msun_to_N_over_A2
            unit_gf_str = "$\,\left[T\\right]$"
        else:
            conv_fac_gf = 1.
            unit_gf_str = ""
            warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'") 


else: raise RuntimeError("Unrecognized units \"" + units + "\"")

################################################################################










############################## PLOT SETUP ######################################

# Initialize the template filename for the plots
figname = plots_dir + "/"

# Initialize some needed lists
simdirs   = []
read_data = []

AHfile_cols1 = []
AHfile_cols2 = []

xlabels = []
ylabels = []
norms   = []

there_are_iters_avail = []
last_valid_it         = []
last_valid_g          = []

xsteps = []
ysteps = []

xmin_plot = []
xmax_plot = []
ymin_plot = []
ymax_plot = []

alpha_grid       = []
alpha_grid_steps = []

alpha_reflevels         = []
alpha_reflevels_steps   = []


for n in range(N_datasets):
    # Add info about grid function and plane to the template filename
    figname += grid_functions[n] + "_" + planes[n] + "_"

    # Build a SimDir object
    simdirs.append(SimDir(data_dirs[n]))


    # Get the correct PostCactus method to read the data on the desired plane,
    # get the correct columns in the AHFinderDirect files to read data from and
    # set the axes' labels
    if (planes[n] == "xy"):
        read_data.append(simdirs[n].grid.xy.read)
        AHfile_cols1.append(3)
        AHfile_cols2.append(4)

        if (input_coords == "Cartesian"):
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("y$\,$" + unit_space_str)
        elif (input_coords == "Exponential fisheye"):
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)

    elif (planes[n] == "xz"):
        read_data.append(simdirs[n].grid.xz.read)
        AHfile_cols1.append(3)
        AHfile_cols2.append(5)

        if (input_coords == "Cartesian"):
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)
        elif (input_coords == "Exponential fisheye"):
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("y$\,$" + unit_space_str)

    elif (planes[n] == "yz"):
        read_data.append(simdirs[n].grid.yz.read)
        AHfile_cols1.append(4)
        AHfile_cols2.append(5)

        if (input_coords == "Cartesian"):
            xlabels.append("y$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME
        elif (input_coords == "Exponential fisheye"):
            xlabels.append("$\\theta$")
            ylabels.append("$\phi$")
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME

    else:
        raise RuntimeError("Unrecognized plane \"" + planes[n] + "\"")


    # Set up the plot scale
    if (logscale[n] == True):
        if (symlogscale[n] == True):
            norms.append(colors.SymLogNorm(vmin      = clb_extents[n][0],
                                           vmax      = clb_extents[n][1],
                                           linthresh = clb_extents[n][0]))
        elif (symlogscale[n] == False):
            norms.append(colors.LogNorm(vmin = clb_extents[n][0],
                                        vmax = clb_extents[n][1]))
        else:
            raise RuntimeError("Please set symlogscale[" + str(n) + "] to either 'True' or 'False'")

    elif (logscale[n] == False):
        if (linscale_norm[n] == "yes"):
            norms.append(colors.Normalize(vmin = clb_extents[n][0],
                                          vmax = clb_extents[n][1]))
        elif (linscale_norm[n] == False):
            norms.append(None)
        else:
            raise RuntimeError("Please set linscale_norm[" + str(n) + "] to either 'True' or 'False'")
    else:
        raise RuntimeError("Please set logscale[" + str(n) + "] to either 'True' or 'False'")


    # Flags to check iteration availability
    there_are_iters_avail.append(True)
    last_valid_it.append(first_it)
    last_valid_g.append(None)


    # Compute the plot size reduction to be applied at each timestep,
    # logarithmically increasing/decreasing the zoom
    if (zooms[n]):
        xmin_init = actual_plot_extents[n][0]
        xmax_init = actual_plot_extents[n][1]
        ymin_init = actual_plot_extents[n][2]
        ymax_init = actual_plot_extents[n][3]

        xmin_end = actual_plot_extents_end[n][0]
        xmax_end = actual_plot_extents_end[n][1]
        ymin_end = actual_plot_extents_end[n][2]
        ymax_end = actual_plot_extents_end[n][3]

        half_range_x_init = 0.5*(xmax_init - xmin_init)
        half_range_y_init = 0.5*(ymax_init - ymin_init)
        assert(half_range_x_init > 0.)
        assert(half_range_y_init > 0.)

        half_range_x_end = 0.5*(xmax_end - xmin_end)
        half_range_y_end = 0.5*(ymax_end - ymin_end)
        assert(half_range_x_end > 0.)
        assert(half_range_y_end > 0.)

        assert(first_its_zoom[n] >= first_it)
        assert( last_its_zoom[n] <=  last_it)
        assert((last_its_zoom[n] - first_its_zoom[n]) % out2D_every == 0)
        n_zoom = int((last_its_zoom[n] - first_its_zoom[n])/out2D_every) + 1

        # If the grid is expanding (half_range_x_end > half_range_x_init), then
        # xsteps and ysteps are monotonically increasing, otherwise they are
        # decreasing
        xsteps.append(np.logspace(np.log(half_range_x_init), np.log(half_range_x_end),
                                  num = n_zoom, base = np.e, endpoint = True))
        ysteps.append(np.logspace(np.log(half_range_y_init), np.log(half_range_y_end),
                                  num = n_zoom, base = np.e, endpoint = True))

    # Initialize the plot range
    if ( input_coords == "Cartesian" or
        (input_coords != "Cartesian" and actual_plot_extents[n] is None)):
        xmin_plot.append(plot_extents[n][0])
        xmax_plot.append(plot_extents[n][1])
        ymin_plot.append(plot_extents[n][2])
        ymax_plot.append(plot_extents[n][3])
    else:  # Non-Cartesian coords, actual_plot_extents[n] is not None
        xmin_plot.append(actual_plot_extents[n][0])
        xmax_plot.append(actual_plot_extents[n][1])
        ymin_plot.append(actual_plot_extents[n][2])
        ymax_plot.append(actual_plot_extents[n][3])


    # If requested, compute the steps over which the grid's transparency should
    # change at every iteration in the assigned range and initialize the grid's
    # transparencies
    if (plot_grid[n]):
        alpha_grid.append(alpha_grid_init[n])
        if (vary_grid_transparency[n]):
            assert((last_its_alpha_grid[n] - first_its_alpha_grid[n]) % out2D_every == 0)
            alpha_grid_n     = (last_its_alpha_grid[n] - first_its_alpha_grid[n])/out2D_every
            alpha_grid_range = alpha_grid_end[n] - alpha_grid_init[n]
            alpha_grid_steps.append(alpha_grid_range/alpha_grid_n)


    # If requested, compute the steps over which the refinement levels'
    # transparency should change at every iteration in the assigned range and
    # initialize the refinement levels' transparencies
    if (plot_reflevels[n]):
        alpha_reflevels.append(alpha_reflevels_init[n])
        if (vary_reflevels_transparency[n]):
            assert((last_its_alpha_reflevels[n] - first_its_alpha_reflevels[n]) % out2D_every == 0)
            alpha_reflevels_n     = (last_its_alpha_reflevels[n] - first_its_alpha_reflevels[n])/out2D_every
            alpha_reflevels_range = alpha_reflevels_end[n] - alpha_reflevels_init[n]
            alpha_reflevels_steps.append(alpha_reflevels_range/alpha_reflevels_n)

################################################################################










################################### PLOT #######################################

# Initialize the frame number
nframe = int(first_it/out2D_every)

# Toy grid used to check whether an iteration is available for a given dataset
xmin_largest = plot_extents[:, 0].max()
xmax_largest = plot_extents[:, 1].max()
ymin_largest = plot_extents[:, 2].max()
ymax_largest = plot_extents[:, 3].max()
g_toy = gd.RegGeom([2, 2], [xmin_largest, ymin_largest],
                      x1 = [xmax_largest, ymax_largest])


# ***** Actually plot the data *****
for it in range(first_it, last_it + 1, out2D_every):
    print("***** Iteration " + str(it) + " *****\n")

    # Can't have 'plt.subplots(1, 1)'
    if (N_datasets > 1):
        fig, axes = plt.subplots(nsubplots_y, nsubplots_x,
                                 figsize = figsize, dpi = dpi)
    else:  # Single subplot -> No arguments
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

    fig.set_tight_layout(True)

    for n in range(N_datasets):
        # The axis object returned by plt.subplots() when there's only one
        # subplot is not subscriptable
        if (N_datasets > 1):
            ax = axes[n]

        # Configure axes
        ax.set_box_aspect(1)  # Square snapshots
        ax.set_title(titles[n], color = titlecolor, y = titlepad,
                     fontsize   = title_fontsize,
                     fontweight = title_fontweight,
                     fontstyle  = title_fontstyle,
                     fontname   = title_fontname)
        ax.set_xlabel(xlabels[n], fontsize = labelsize, labelpad = labelpad_x)
        ax.set_ylabel(ylabels[n], fontsize = labelsize, labelpad = labelpad_y)
        ax.tick_params(labelsize = ticksize)
        ax.set_xlim(xmin_plot[n], xmax_plot[n])
        ax.set_ylim(ymin_plot[n], ymax_plot[n])

        # Try to read data on a small, "toy" grid just to make sure the current
        # iteration is available for the current dataset
        patch_toy = read_data[n](grid_functions[n], it,
                                 geom           = g_toy,
                                 adjust_spacing = True,
                                 order          = 0,
                                 outside_val    = 0.,
                                 level_fill     = False)


        # If iteration 'it' is not available for this dataset, then flag it. If
        # all the other datasets are flagged already, then it means there's
        # nothing else to be plotted: break. Otherwise, reset to the last
        # non-flagged iteration for this dataset.
        if (patch_toy.time is None):
            print("Setting dataset " + str(n) + " to the last valid iteration")
            there_are_iters_avail[n] = False
            no_iters_avail_count     = 0

            for avail in there_are_iters_avail:
               if (not avail): no_iters_avail_count += 1

            assert(no_iters_avail_count <= N_datasets)

            if (no_iters_avail_count == N_datasets):
                # Dirty hack to break a nested loop
                raise StopIteration

            else: 
                patch_plot = read_data[n](grid_functions[n], last_valid_it[n],
                                          geom           = last_valid_g[n],
                                          adjust_spacing = True,
                                          order          = 0,
                                          outside_val    = 0.,
                                          level_fill     = True)

        else:
            # Build an object containing the grid hierarchy, i.e. a list of
            # objects each containing information about the grid patch (size,
            # resolution, time, iteration, refinement level, number of ghost
            # cells, etc.) and the associated data 
            patches = read_data[n](grid_functions[n], it,
                                   geom           = None,
                                   adjust_spacing = True,
                                   order          = 0,
                                   outside_val    = 0.,
                                   level_fill     = False)

            # Find the grid spacing on the finest refinement level and, if
            # desired, draw the refinement level contours and save them for the
            # next iteration
            for i in range(len(patches)):
                geom   = patches[i].geom()
                deltas = geom.dx()
                dx     = deltas[0]
                dy     = deltas[1]

                if (i == 0):
                    deltax_min = dx
                    deltay_min = dy
                else:
                    if (dx < deltax_min): deltax_min = dx
                    if (dy < deltay_min): deltay_min = dy

            print("Dataset " + str(n) + ": finest grid spacing is ("
                  + str(deltax_min) + ", " + str(deltay_min) + ")")


            # Set the geometry of the grid to be plotted
            xmin  = plot_extents[n][0]
            xmax  = plot_extents[n][1]
            ymin  = plot_extents[n][2]
            ymax  = plot_extents[n][3]

            Nx = int((xmax - xmin)/deltax_min)
            Ny = int((ymax - ymin)/deltay_min)

            g = gd.RegGeom([Nx, Ny], [xmin, ymin], x1 = [xmax, ymax])


            # Build the patch to be plotted, which is resampled to a uniform grid
            # (with the finest grid spacing)
            patch_plot = read_data[n](grid_functions[n], it,
                                      geom           = g,
                                      adjust_spacing = True,
                                      order          = 0,
                                      outside_val    = 0.,
                                      level_fill     = False)


        # The option 'adjust_spacing = True' above may reshape patch_plot.data
        # in order to snap to the finest resolution available, so that
        # np.transpose(patch_plot.data) may not have dimensions (Nx, Ny)
        # anymore. However, (almost) no reshaping shoud happen here because Nx
        # and Ny were determined by the finest resolution to begin with. In any
        # event, reset Nx and Ny to their new values.
        # NOTE: alternatively, one could specify 'adjust_spacing = False' and
        # PostCactus wouldn't snap to the finest available resolution, leaving
        # the shape of patch_plot untouched.
        if (abs_vals[n]):
            plot_data = np.absolute(patch_plot.data*conv_fac_gf)
        else:
            plot_data = patch_plot.data*conv_fac_gf

        Nx_new = plot_data.shape[0]
        Ny_new = plot_data.shape[1]

        if (Nx_new != Nx or Ny_new != Ny):
            print("Dataset " + str(n) + ": grid reshaped from (" + str(Nx) + ", " + str(Ny) + ") to (" + str(Nx_new) + ", " + str(Ny_new) + ")")


        # Build the mesh for pcolormesh and transform to Cartesian coordinates
        # if needed
        # NOTE: there are Nx cells => Nx+1 edges (and the same for Ny)
        if (input_coords == "Cartesian"):
            x       = np.linspace(xmin, xmax, Nx_new)
            y       = np.linspace(ymin, ymax, Ny_new)
            mx, my  = np.meshgrid(x, y)
            mx_plot = mx
            my_plot = my
        elif (input_coords == "Exponential fisheye"):
            logr        = np.linspace(xmin, xmax, Nx_new)
            phi         = np.linspace(ymin, ymax, Ny_new)
            mlogr, mphi = np.meshgrid(logr, phi)
            mx          = np.exp(mlogr)*np.cos(mphi)
            my          = np.exp(mlogr)*np.sin(mphi)
            mx_plot     = mx
            my_plot     = my
        else:
            raise RuntimeError("Invalid input coordinates")


        # Build the image to plot
        # NOTE: since mxcoords, mycoords and np.transpose(plot_data) all have
        #       the same shape, shading = "auto" should produce
        #       shading = "nearest", meaning that data are cell-centered. On the
        #       other hand, if mxcoords and mycoords had one point more than
        #       np.transpose(plot_data) in each direction, than the data would
        #       be placed on cell vertices and shading = "auto" should produce
        #       shading = "flat".
        if (plot_grid[n]):
            im = ax.pcolormesh(mx, my, np.transpose(plot_data),
                               shading = "auto", cmap = cmaps[n], norm = norms[n],
                               edgecolor = (0., 0., 0., alpha_grid[n]),
                               linewidth = 0.01)
        else:
            im = ax.pcolormesh(mx, my, np.transpose(plot_data),
                               shading = "auto", cmap = cmaps[n], norm = norms[n])

        # Add a colorbar
        if (add_clb[n]):
            clb = fig.colorbar(im, ax = ax, extend = clb_extends[n],
                               fraction = clb_fraction)
            clb.ax.set_title(varnames[n] + unit_gf_str, pad = clblabel_pad,
                             fontsize   = clblabel_fontsize,
                             fontweight = clblabel_fontweight,
                             fontstyle  = clblabel_fontstyle)
            clb.ax.tick_params(labelsize = clb_ticksize)


        # Plot apparent horizons by finding the convex hull of the projection of
        # the points defining it (as found by AHFinderDirect) in the desired
        # plane
        if (draw_AH[n]):
            print("Dataset " + str(n) + ": trying to draw apparent horizon(s)...")
            found_AH_data = False

            for r in range(1, N_AH_files + 1):
                if (there_are_iters_avail[n]):
                    AH_file = AH_dirs[n] + "/h.t" + str(it) + ".ah" + str(r) + ".gp"
                else:
                    AH_file = AH_dirs[n] + "/h.t" + str(int(last_valid_it[n])) + ".ah" + str(r) + ".gp"

                # In case there are multiple files available for the same
                # horizon, the following overrides the previous data
                if (path.exists(AH_file)):
                    found_AH_data = True
                    AH_data       = np.loadtxt(AH_file)
                    hull          = ConvexHull(AH_data[:, [AHfile_cols1[n], AHfile_cols2[n]]])
                    xhull         = AH_data[:, AHfile_cols1[n]][hull.vertices]
                    yhull         = AH_data[:, AHfile_cols2[n]][hull.vertices]

                    ax.fill(xhull*conv_fac_space, yhull*conv_fac_space,
                            linewidth = 0., facecolor = "black")

                    print("Dataset " + str(n) + ": apparent horizon " + str(r) + " drawn from AH file " + str(r))

            print("")

            if (not found_AH_data):
                warnings.warn("Dataset " + str(n) + ": no AH data found")


        # Compute the min and max value in the plotted data if requested
        """
        if (compute_min_max[n]):
            if (abs_vals[n]):
                abs_data = np.absolute(patch_plot.data)
                minval = abs_data.min()
                maxval = abs_data.max()
            else:
                minval = patch_plot.data.min()
                maxval = patch_plot.data.max()

            print("Dataset " + str(n) + ": min = " + str(minval) + ", max = " + str(maxval))

            # FIXME FIXME FIXME FIXME FIXME FIXME
            # Fix the position of the min/max
            # FIXME FIXME FIXME FIXME FIXME FIXME
            fig.text(0.15, 0.2, "Min: " + str(minval),
                     fontsize = 17., fontweight = "bold",
                     fontstyle = myfontstyle)
            fig.text(0.15, 0.17, "Max: " + str(maxval),
                     fontsize = 17., fontweight = "bold",
                     fontstyle = myfontstyle)
        """


        # Zoom in/out if required and reset the plot range if necessary
        if (zooms[n] and it >= first_its_zoom[n]
                     and it <  last_its_zoom[n]):
            assert((it - first_its_zoom[n]) % out2D_every == 0)
            snapshot_index = int((it - first_its_zoom[n])/out2D_every)
            assert(snapshot_index >= 0 and snapshot_index < xsteps[n].shape[0]
                                       and snapshot_index < ysteps[n].shape[0])

            dlog_x = xsteps[n][snapshot_index + 1] - xsteps[n][snapshot_index]
            dlog_y = ysteps[n][snapshot_index + 1] - ysteps[n][snapshot_index]

            # If the grid is expanding (half_range_x_end > half_range_x_init),
            # then xsteps and ysteps are monotonically increasing and so
            # dlog_x > 0; otherwise dlog_x < 0 (same for dlog_y)
            xmin_plot[n] -= dlog_x
            xmax_plot[n] += dlog_x
            ymin_plot[n] -= dlog_y
            ymax_plot[n] += dlog_y


        # Reset the grid's transparency if desired
        if (plot_grid[n] and vary_grid_transparency[n] and
            it >= first_its_alpha_grid[n] and
            it <   last_its_alpha_grid[n]):
            alpha_grid[n] += alpha_grid_steps[n]


        # Plot the refinement level boundaries if desired
        if (plot_reflevels[n]):
            patch_contour = read_data[n](grid_functions[n], it,
                                         geom           = g,
                                         adjust_spacing = True,
                                         order          = 0,
                                         outside_val    = 0.,
                                         level_fill     = True)
            levels = np.arange(reflevel_ranges[n][0], reflevel_ranges[n][1] + 1)
            reflevels_colors = []
            for i in range(len(levels)):
                reflevels_colors.append((0., 0., 0., alpha_reflevels[n]))
            viz.plot_contour(patch_contour, levels = levels,
                             colors = reflevels_colors)

            # Reset the reflevels' transparency if desired 
            if (vary_reflevels_transparency[n] and
                it >= first_its_alpha_reflevels[n] and
                it <   last_its_alpha_reflevels[n]):
                alpha_reflevels[n] += alpha_reflevels_steps[n]


        # Reset the last valid iteration and the geometry if needed
        if (there_are_iters_avail[n]):
            last_valid_it[n] = it
            last_valid_g[n]  = g


    # Time and iteration info
    it_str   = "It = " + str(it)
    ##time_str = "t = " + str("{:.2e}".format(patch_plot.time*conv_fac_time))
    time_str = "t = " + str("{:.2e}".format((t0 + patch_plot.time)*conv_fac_time))

    fig.text(it_pos[0], it_pos[1], it_str, color = "red",
             fontsize  = it_time_fontsize,  fontweight = it_time_fontweight,
             fontstyle = it_time_fontstyle)
    fig.text(time_pos[0], time_pos[1], time_str + unit_time_str, color = "red",
             fontsize  = it_time_fontsize,  fontweight = it_time_fontweight,
             fontstyle = it_time_fontstyle)


    # Get ready for the next iteration
    plt.savefig(figname + str("{:0>4d}".format(nframe)) + fig_ext)
    plt.close()
    nframe += 1
    print("Done\n\n\n")
