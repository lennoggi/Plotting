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
from postcactus.simdir import SimDir
from postcactus import grid_data as gd
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
    "/lagoon/bbhdisk/CBD_SphericalNR/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP"
])


# Directory where the plots will be placed
# ----------------------------------------
plots_dir = "/lagoon/lennoggi/Snapshots/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP"


# File extension for the plots
# ----------------------------
fig_ext = ".png"


# Which grid functions to plot
# ----------------------------
grid_functions = np.array([
    "rho"
])


# Input coordinates
# -----------------
input_coords = "Exponential fisheye"


# Plot absolute values?
# ---------------------
abs_vals = np.array([
    False
])


# Which 2D slices to plot
# Cartesian coordinates: xy, xz or yz plane
# Spherical coordinates: r-theta, r-phi or theta-phi plane
# --------------------------------------------------------
planes = np.array([
    "xz"
])


# Plot extent **IN NUMERICAL COORDINATES**; provide it as
# [xmin, xmax, ymin, ymax]. Used to build the PostCactus geometry.
# ----------------------------------------------------------------
plot_extents = np.array([
     np.array([np.log(15.), np.log(2000.), 0., 2.*np.pi])
])


# Actual plot extent if the input coordinates are not Cartesian, i.e., what you
# will actually see in the snapshot(s)
# NOTE: set to None if you want to keep the original grid dimensions when using
#       non-Cartesian coordinate systems
actual_plot_extent = np.array([
    np.array([-300., 300., -300., 300.])
])


# Which iterations to plot
# ------------------------
first_it    = 0
last_it     = 1 ##1000000000  # Set this to a huge number to plot all iterations
out2D_every = 400 ##512


# Apparent horizon
# ----------------
draw_AH = np.array([
    False
])

# How many files per AH are there, i.e. the maximum value of 'AH_number' in
# 'h.t<iteration>.ah<AH_number>'
# -------------------------------------------------------------------------
N_AH_files = 2

# **IMPORTANT**
# Make sure all AH files ('h.t<iteration>.ah<AH_number>') are in the same
# directory. In case they live under different 'output-xxxx' directories, copy
# them to a common directory
# # --------------------------------------------------------------------------
AH_dirs = np.array([
##    "/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/AH_data",
    "/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/AH_data"
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


# Add colorbars?
# --------------
add_colorbar = np.array([
    True
])


# Title and other text options
# ----------------------------
titlecolor       = "midnightblue"
titlepad         = 1.02
title_fontsize   = 35.
title_fontweight = "bold"
title_fontstyle  = "normal"
title_fontname   = "Ubuntu"

labelsize  = 25.
labelpad_x = 3.
labelpad_y = -5.
ticksize   = 20.

clb_ticksize        = 20.
clblabel_pad        = 8.
clb_fraction        = 0.05
clblabel_fontsize   = 25.
clblabel_fontweight = "bold"
clblabel_fontstyle  = "normal"

it_pos             = np.array([0.15, 0.015])
time_pos           = np.array([0.6, 0.015])
it_time_fontsize   = 25.
it_time_fontweight = "bold"
it_time_fontstyle  = "normal"


# Subplots layout
# ---------------
nsubplots_x = 1
nsubplots_y = 1

# Figure size and resolution
# --------------------------
figsize = [12., 10.]
dpi     = 200


# Extent of the color scales (note that the actual scale may extend below
# colorbar_extents[i][0] if logscale[i] = "yes" and symlogscale[i] 0 "yes")
# -------------------------------------------------------------------------
colorbar_extents = np.array([
    np.array([1.e-08, 1.5e-02])
])


# Logarithmic scale
# -----------------
logscale = np.array([
    True
])


# Symmetric logarithmic scale: if a logarithmic scale is in use and data values
# extend down to zero, then a linear scale is used from zero to the desired
# minimum in the colorbar
# -----------------------------------------------------------------------------
symlogscale = np.array([
    False
])


# Normalize the linear color scale between 0 and 1 (only if a logarithmic scale
# is not in use)
# -----------------------------------------------------------------------------
linscale_norm = np.array([
    True
])


# Colormap
# --------
cmaps = np.array([
    "plasma"
])


# Type of colorbar extension outside its limits ("neither", "max", "min" or
# "both")
# -------------------------------------------------------------------------
clb_extend = np.array([
    "max"
])


# Choose if you want to find max and min in the data for every iteration
# available or not (this may take some time)
# ----------------------------------------------------------------------
compute_min_max = np.array([
    False
])

################################################################################










######################### CODE SELF-PROTECTION #################################

assert(first_it >= 0)
assert(last_it  >= first_it)

N_datasets = len(data_dirs)
assert(nsubplots_x*nsubplots_y == N_datasets)

assert(input_coords == "Cartesian" or
       input_coords == "Exponential fisheye")

assert(len(grid_functions)   == N_datasets)
assert(len(abs_vals)         == N_datasets)
assert(len(planes)           == N_datasets)
assert(len(plot_extents)     == N_datasets)
assert(len(draw_AH)          == N_datasets)
assert(len(AH_dirs)          == N_datasets)
assert(len(varnames)         == N_datasets)
assert(len(titles)           == N_datasets)
assert(len(colorbar_extents) == N_datasets)
assert(len(logscale)         == N_datasets)
assert(len(symlogscale)      == N_datasets)
assert(len(linscale_norm)    == N_datasets)
assert(len(cmaps)            == N_datasets)
assert(len(clb_extend)       == N_datasets)
assert(len(compute_min_max)  == N_datasets)

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

# Initialize some needed arrays
simdirs               = []
read_data             = []
AHfile_cols1          = []
AHfile_cols2          = []
xlabels               = []
ylabels               = []
norms                 = []
there_are_iters_avail = []
last_valid_it         = []
last_valid_g          = []


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
            norms.append(colors.SymLogNorm(vmin      = colorbar_extents[n][0],
                                           vmax      = colorbar_extents[n][1],
                                           linthresh = colorbar_extents[n][0]))
        elif (symlogscale[n] == False):
            norms.append(colors.LogNorm(vmin = colorbar_extents[n][0],
                                        vmax = colorbar_extents[n][1]))
        else:
            raise RuntimeError("Please set symlogscale[" + str(n) + "] to either 'True' or 'False'")

    elif (logscale[n] == False):
        if (linscale_norm[n] == "yes"):
            norms.append(colors.Normalize(vmin = colorbar_extents[n][0],
                                          vmax = colorbar_extents[n][1]))
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
for it in range(first_it, last_it, out2D_every):
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
                                          level_fill     = False)

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

            # Find the grid spacing on the finest refinement level
            for i in range(len(patches)):
                deltas = patches[i].geom().dx()
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
        if (abs_vals[n]):
            im = ax.pcolormesh(mx, my, np.absolute(np.transpose(plot_data)),
                               shading = "auto", cmap = cmaps[n], norm = norms[n])
        else:
            im = ax.pcolormesh(mx, my, np.transpose(plot_data),
                               shading = "auto", cmap = cmaps[n], norm = norms[n])

        # Add a colorbar
        if (add_colorbar[n]):
            clb = fig.colorbar(im, ax = ax, extend = clb_extend[n],
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


        # Reset the plot dimensions if the coordinates are not Cartesian and if
        # desired
        if (input_coords != "Cartesian" and actual_plot_extent is not None):
            ax.set_xlim(actual_plot_extent[n][0], actual_plot_extent[n][1])
            ax.set_ylim(actual_plot_extent[n][2], actual_plot_extent[n][3])


        # Reset the last valid iteration and the geometry if needed
        if (there_are_iters_avail[n]):
            last_valid_it[n] = it
            last_valid_g[n]  = g


    # Time and iteration info
    it_str   = "It = " + str(it)
    time_str = "t = " + str("{:.2e}".format(patch_plot.time*conv_fac_time))

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
