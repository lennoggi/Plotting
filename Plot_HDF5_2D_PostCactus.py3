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
data_dirs = [
    "/scratch1/07825/lennoggi/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M",
    "/scratch3/07825/lennoggi/CBD_HydroDiskID_IGM_McLachlan_Spinning_aligned08_OrbSep10M"
]


# Directory where the plots will be placed
# ----------------------------------------
plots_dir = "/scratch1/07825/lennoggi/Snapshots/Comparisons/CBD_SpinningAligned08_EqDiskVsHydroDiskID_OrbSep10M/radcool_gf_xy"


# File extension for the plots
# ----------------------------
fig_ext = ".png"


# Which grid functions to plot
# ----------------------------
grid_functions = [
    "radcool_gf", ##"rho_b",
    "radcool_gf"  ##"rho_b"  ##"smallb2"
]


# Plot absolute values?
# ---------------------
abs_vals = [
    False,
    False
]


# Which 2D slices to plot: xy, xz or yz plane
# -------------------------------------------
planes = [
    "xy",
    "xy"
]


# Plot extent; give it as [xmin, xmax, ymin, ymax]
# ------------------------------------------------
plot_extents = [
    [-40., 40., -40., 40.],
    [-40., 40., -40., 40.]
]


# Which iterations to plot
# ------------------------
first_it    = 0 
last_it     = 1000000000  # Set this to a huge number to plot all iterations
out2D_every = 1024


# Apparent horizon
# ----------------
draw_AH = [
    True,
    True
]

# How many files per AH are there, i.e. the maximum value of 'AH_number' in
# 'h.t<iteration>.ah<AH_number>'
N_AH_files = 2

# **IMPORTANT**
# Make sure all AH files ('h.t<iteration>.ah<AH_number>') are in the same
# directory. In case they live under different 'output-xxxx' directories, copy
# them to a common directory
AH_dirs = [
    "/scratch1/07825/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/AH_data",
    "/scratch1/07825/lennoggi/Snapshots/CBD_HydroDiskID_IGM_McLachlan_Spinning_aligned08_OrbSep10M/AH_data"
]





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
varnames = [
    "       $\\rho$",
    "       $\\rho$"  ##"       b^2"
]


# Titles for each subplot
# -----------------------
titles = [
    "Steadily accreting CBD",
    "Torus"
]


# Title and other text options
# ----------------------------
titlecolor   = "midnightblue"
myfontweight = "bold"
myfontfamily = "sans-serif"
myfontstyle  = "normal"
##myfontname   = "Ubuntu"


# Coordinates of the axes of each subplot and colorbar
# ----------------------------------------------------
# ***** Suggested options *****
# 1. Single plot with small colorbar on the left
#    axplots = [
#        [0.15, 0.09, 0.83, 0.83]
#    ]
#    axclbs = [
#        [0.03, 0.6, 0.03, 0.32]
#    ]
#
# 2. Single plot with large colorbar on the right (set figsize = 20., 17], dpi = 200)
#    axplots = [
#        [0.08, 0.12, 0.8, 0.8]
#    ]
#    axclbs = [
#        [0.84, 0.12, 0.04, 0.8]
#    ]
#
# 3. Two plots with one colorbar each
#    axplots = [
#        [-0.115, 0.13, 0.78, 0.78],
#        [0.405, 0.13, 0.78, 0.78]
#    ]
#    axclbs = [
#        [0.005, 0.13, 0.015, 0.78],
#        [0.525, 0.13, 0.015, 0.78]
#    ]
# 4. Two plots with a single colorbar
#    TODO
# 5. TODO
axplots = [
    [-0.115, 0.13, 0.78, 0.78],
    [0.405, 0.13, 0.78, 0.78]
]

axclbs = [
    [0.005, 0.13, 0.015, 0.78],
    [0.525, 0.13, 0.015, 0.78]
]


# Extent of the color scales (note that the actual scale may extend below
# colorbar_extents[i][0] if logscale[i] = "yes" and symlogscale[i] 0 "yes")
# -----------------------------------------------------------------------
colorbar_extents = [
    [1.e-15, 1.e-07], ##[1.e-08, 1.5e-02],
    [1.e-15, 1.e-07]  ##[1.e-08, 1.5e-02] ##[4.e-15, 4.e-7]
]


# Logarithmic scale
# -----------------
logscale = [
    True,
    True
]


# Symmetric logarithmic scale: if a logarithmic scale is in use and data values
# extend down to zero, then a linear scale is used from zero to the desired
# minimum in the colorbar
# -----------------------------------------------------------------------------
symlogscale = [
    True,
    True
]


# Normalize the linear color scale between 0 and 1 (only if a logarithmic scale
# is not in use)
# -----------------------------------------------------------------------------
linscale_norm = [
    True,
    True
]


# Colormap
# --------
cmaps = [
    "plasma",
    "plasma" ##viridis"
]


# Type of colorbar extension outside its limits ("neither", "max", "min" or
# "both")
clb_extend = [
    "both", ##"max""
    "both"  ##"max""
]


# Choose if you want to find max and min in the data for every iteration
# available or not (this may take some time)
compute_min_max = [
    False,
    False
]

################################################################################










######################### CODE SELF-PROTECTION #################################

assert(first_it >= 0)
assert(last_it  >= first_it)

N_datasets = len(data_dirs)

assert(len(grid_functions)   == N_datasets)
assert(len(abs_vals)         == N_datasets)
assert(len(planes)           == N_datasets)
assert(len(plot_extents)     == N_datasets)
assert(len(draw_AH)          == N_datasets)
assert(len(AH_dirs)          == N_datasets)
assert(len(varnames)         == N_datasets)
assert(len(titles)           == N_datasets)
assert(len(axplots)          == N_datasets)
assert(len(axclbs)           == N_datasets)
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

# Initialize some needed lists
simdirs     = []
read_data   = []
plot_extent = []
norms       = []
xlabels     = []
ylabels     = []


for n in range(N_datasets):
    # Add info about grid function and plane to the template filename
    figname += grid_functions[n] + "_" + planes[n] + "_"

    # Build a SimDir object
    sd = SimDir(data_dirs[n])
    simdirs.append(sd)

    # Get the correct PostCactus method to read the data on the desired plane
    # and the correct columns in the AHFinderDirect fils to read data from 
    if (planes[n] == "xy"):
        read_data.append(sd.grid.xy.read)
        column_1 = 3
        column_2 = 4
    elif (planes[n] == "xz"):
        read_data.append(sd.grid.xz.read)
        column_1 = 3
        column_2 = 5
    elif (planes[n] == "yz"):
        read_data.append(sd.grid.yz.read)
        column_1 = 4
        column_2 = 5
    else: raise RuntimeError("Unrecognized plane \"" + planes[n] + "\"")


    # Set up the plot scale
    if (logscale[n] == True):
        if (symlogscale[n] == True):
            norms.append(colors.SymLogNorm(vmin      = colorbar_extents[n][0],
                                           vmax      = colorbar_extents[n][1],
                                           linthresh = colorbar_extents[n][0]))
        else:
            norms.append(colors.LogNorm(vmin = colorbar_extents[n][0],
                                        vmax = colorbar_extents[n][1]))
    else:
        if (linscale_norm[n] == "yes"):
            norms.append(colors.Normalize(vmin = colorbar_extents[n][0],
                                          vmax = colorbar_extents[n][1]))
        else:
            norms.append(None)


    # Set axes labels
    xlabels.append(planes[n][0] + "$\,$" + unit_space_str)
    ylabels.append(planes[n][1] + "$\,$" + unit_space_str)

################################################################################










################################### PLOT #######################################

# Initialize the frame number
nframe = int(first_it/out2D_every)


# Flags to tell whether there are still iterations available for plotting and
# last valid iterations to which the plots are reset in case there are no more
# iterations to plot
there_are_iters_avail = np.empty(N_datasets)
last_valid_it         = np.empty(N_datasets)
last_valid_g          = []

for n in range(N_datasets):
    there_are_iters_avail[n]  = True
    last_valid_it[n]          = first_it
    last_valid_g.append(None)


# Small "toy" grid used to check whether an iteration is available for a given
# dataset
g_toy = gd.RegGeom([2, 2], [0., 0.], x1 = [1., 1.])



# ***** Actually plot the data *****
for it in range(first_it, last_it, out2D_every):
    print("***** Iteration " + str(it) + " *****\n")

    fig = plt.figure(figsize = [20., 10.], dpi = 100)  ##[10., 10.], dpi = 200  ##[20., 17.], dpi = 200)

    for n in range(N_datasets):
        # Configure axes
        axplot = fig.add_axes(axplots[n])
        axplot.set_title(titles[n], y = 1.01, fontsize = 30., fontweight = "bold",
                         fontfamily = myfontfamily, fontstyle = myfontstyle, ##fontname = myfontname,
                         color = titlecolor)
        axplot.set_xlabel(xlabels[n], fontsize = 20., labelpad = 8.)
        axplot.set_ylabel(ylabels[n], fontsize = 20., labelpad = -6.)
        ##axplot.tick_params(labelsize = 70.)

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

                if (i == 0): delta_min = min(dx, dy)

                else:
                    if (dx < delta_min): delta_min = dx
                    if (dy < delta_min): delta_min = dy

            delta_min = 0.25  # FIXME: remove
            print("Dataset " + str(n) + ": smallest grid spacing is " + str(delta_min))


            # Set the geometry of the grid to be plotted
            min1 = plot_extents[n][0]
            max1 = plot_extents[n][1]
            min2 = plot_extents[n][2]
            max2 = plot_extents[n][3]

            N1 = int(max1 - min1)/delta_min
            N2 = int(max2 - min2)/delta_min
            g  = gd.RegGeom([N1, N2], [min1, min2], x1 = [max1, max2])


            # Build the patch to be plotted, which is resampled to a uniform grid
            # (with the finest grid spacing)
            patch_plot = read_data[n](grid_functions[n], it,
                                      geom           = g,
                                      adjust_spacing = True,
                                      order          = 0,
                                      outside_val    = 0.,
                                      level_fill     = False)


        # Build the image to plot
        if (abs_vals[n]):
            im = axplot.imshow(np.transpose(np.absolute(patch_plot.data*conv_fac_gf)),
                               cmap   = cmaps[n],        origin = "lower",
                               extent = plot_extents[n], norm   = norms[n])
        else:
            im = axplot.imshow(np.transpose(patch_plot.data*conv_fac_gf),
                               cmap   = cmaps[n],        origin = "lower",
                               extent = plot_extents[n], norm   = norms[n])


        # Add a colorbar
        axclb = fig.add_axes(axclbs[n])
        ##axclb.tick_params(labelsize = 70.)
        clb = fig.colorbar(im, cax = axclb, extend = clb_extend[n],
                           orientation = "vertical")
        clb.ax.set_title(varnames[n] + unit_gf_str, fontsize  = 20., ##70.,
                         fontweight = myfontweight, fontstyle = myfontstyle,
                         fontfamily = myfontfamily, pad = 10.) ##, fontname   = myfontname)


        # Plot apparent horizons by finding the convex hull of the projection of
        # the points defining it (as found by AHFinderDirect) in the xy plane
        if (draw_AH[n]):
            print("Dataset " + str(n) + ": trying to draw apparent horizon(s)...")
            found_AH_data = False

            for r in range(1, N_AH_files + 1):
                if (there_are_iters_avail[n]):
                    AH_file = AH_dirs[n] + "/h.t" + str(it)               + ".ah" + str(r) + ".gp"
                else:
                    AH_file = AH_dirs[n] + "/h.t" + str(int(last_valid_it[n])) + ".ah" + str(r) + ".gp"

                # In case there are multiple files available for the same
                # horizon, the following overrides the previous data
                if (path.exists(AH_file)):
                    found_AH_data = True
                    AH_data       = np.loadtxt(AH_file)
                    hull          = ConvexHull(AH_data[:, [column_1, column_2]])
                    xhull         = AH_data[:, column_1][hull.vertices]
                    yhull         = AH_data[:, column_2][hull.vertices]

                    axplot.fill(xhull*conv_fac_space, yhull*conv_fac_space,
                                linewidth = 0., facecolor = "black")

                    print("Dataset " + str(n) + ": apparent horizon " + str(r) + " drawn from AH file " + str(r))

            print("")

            if (not found_AH_data):
                warnings.warn("Dataset " + str(n) + ": no AH data found")


        # Compute the min and max value in the plotted data if requested
        if (compute_min_max[n]):
            if (abs_vals[n]):
                abs_data = np.absolute(patch_plot.data)
                minval = abs_data.min()
                maxval = abs_data.max()
            else:
                minval = patch_plot.data.min()
                maxval = patch_plot.data.max()

            print("Dataset " + str(n) + ": min = " + str(minval) + ", max = " + str(maxval))

            # FIXME: fix the position of the min/max
            fig.text(0.2, 0.2, "Min: " + str(minval),
                     fontsize = 25., fontweight = "bold",
                     fontstyle = myfontstyle,
                     fontfamily = myfontfamily) ##, fontname = myfontname)
            fig.text(0.2, 0.17, "Max: " + str(maxval),
                     fontsize = 25., fontweight = "bold",
                     fontstyle = myfontstyle,
                     fontfamily = myfontfamily) ##, fontname = myfontname)



        # Reset the last valid iteration and the geometry if needed
        if (there_are_iters_avail[n]):
            last_valid_it[n] = it
            last_valid_g[n]  = g


        # Time and iteration info
        it_str   = "It = " + str(it)
        time_str = "t = " + str("{:.2e}".format(patch_plot.time*conv_fac_time))

        fig.text(0.35, 0.03, it_str, color = "red", fontsize = 25.,
                 fontweight = "bold", fontstyle = myfontstyle,
                 fontfamily = myfontfamily) ##, fontname = myfontname)
        fig.text(0.6, 0.03, time_str + unit_time_str, color = "red",
                 fontsize = 25., fontweight = "bold",
                 fontstyle = myfontstyle,
                 fontfamily = myfontfamily) ##, fontname = myfontname)


    # Get ready for the next iteration
    plt.savefig(figname + str("{:0>4d}".format(nframe)) + fig_ext)
    plt.close()
    nframe += 1
    print("Done\n\n\n")
