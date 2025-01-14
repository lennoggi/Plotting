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
from postcactus import visualize as viz
from scipy.spatial import ConvexHull
import os
import warnings

from matplotlib import rc
from matplotlib import rcParams
##rc("mathtext", usetex = True)
##rc("font",**{"family":"sans-serif","sans-serif":["Avant Garde"]})
##rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
rcParams["mathtext.fontset"] = "dejavuserif"




# ==========
# Parameters
# ==========

# ----------------
# General settings
# ----------------

# Directories containing the files to be opened
datadirs = (
    ##"/scratch3/07825/lennoggi/CBD_prod_MPWZ9_724_140_280_rmin_15_rmax_2e4_q_1_d_20_NZ_FZ/output-0012/data",
    ##"/scratch3/07825/lennoggi/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling",
    ##"/scratch3/07825/lennoggi/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling"
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling_NewCooling/output-0016/HDF5_2D",
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling_NewCooling/output-0016/HDF5_2D"
)

# Simulation restarts to be skipped (set to 'None' to go over all of them)
skip_restarts = None
##skip_restarts = (
##    {"output-0003", "output-0019", "output-0020"},
##    {"output-0003", "output-0019", "output-0020"}
##)

# Directory where the plots will be placed
##plotdir = "/scratch3/07825/lennoggi/Movies/CBD_prod_MPWZ9_724_140_280_rmin_15_rmax_2e4_q_1_d_20_NZ_FZ"
plotdir = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling_NewCooling/rho_b_xy_rho_b_xz"


# Which grid functions to plot as field variables
gfs = (
    ##"rho",
    "rho_b",
    "rho_b"
)

# Which grid functions to plot as stream variables, i.e. integral curves of a
# vector field ('None' plots nothing)
gfs_stream = None
##gfs_stream = (
##    ("Bx", "By"),
##    ##("Bx", "Bz")
##)

# Streamline density in plt.streamline (default in Matplotlib is 1)
streamline_density = 4.

# Input coordinates
input_coords = "Cartesian"  # "Cartesian" or "Exponential fisheye"

# Which 2D slices to plot
#   - Cartesian      coordinates: xy, xz or yz plane
#   - Spherical-like coordinates: r-theta, r-phi or theta-phi plane
planes = (
    ##"xz",
    "xy",
    "xz"
)

# Plot absolute values?
abs_vals = (
    False,
    False
)


# Iterations and initial time info
first_it    = 1368064 ##1540096 ##0
last_it     = 1000000000  # Set this to a huge number to plot all iterations
out2D_every = 1024 ##400
t0          = 0.

# Binary orbit counting
orb_count = False
omega     = 1.049229e-02

# FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
# Do you want to find the max and min in the data for every available
# iteration?
# NOTE: this may take some time
compute_min_max = (
    False,
    False
)
# FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME



# --------------------------
# Figure and layout settings
# --------------------------

# Plot extent **IN NUMERICAL COORDINATES**; provide it as
# (xmin, xmax, ymin, ymax). Used to build the PostCactus geometry.
# NOTE: np.array needed to facilitate slicing
plot_extents = np.array([
     ##np.array([np.log(15.1), np.log(60.),  0., 2.*np.pi]),  #  40*sqrt(2) = 56.5685424949, 60 > 56.5685424949
     np.array([-40.,  40.,  -40.,  40.]),
     np.array([-200., 200., -200., 200.])
])

# Actual plot extent if the input coordinates are not Cartesian, i.e., what you
# will actually see in the snapshot(s)
# NOTE: set to None if you want to keep the original grid dimensions when using
#       non-Cartesian coordinate systems
# NOTE: used as the starting plot extent if 'zoom' (see below) is True
##actual_plot_extents = None
actual_plot_extents = (
    (-40.,  40.,  -40.,  40.),
    (-200., 200., -200., 200.)
)

# Subplots layout
nsubplots_x = 2 ##1
nsubplots_y = 1

# Figure size and resolution
##figsize = [12., 10.]  # Single frame with colorbar on the right
##dpi     = 200
figsize = [22., 10.]  # Two frames with one colorbar only (on the far right)
dpi     = 100
##figsize = [24., 10.]  # Two frames with one colorbar each
##dpi     = 100
##figsize = [33., 10.]  # Three frames with one colorbar only (on the far right)
##dpi     = 100

# File extension for the plots
fig_ext = ".png" ##".jpg"

# Limit resolution (saves memory and time)?
limit_resolution = (
    False,
    True
)

# Resolution to be used if limit_resolution is True
resolution = (
    0.125,
    0.125
)



# -------------------------
# Apparent horizon settings
# -------------------------

# Draw the apparent horizon(s)?
draw_AH = (
    True,
    True
)

# How many AH files there are per iteration, i.e. the maximum value of
# 'AH_number' in 'h.t<iteration>.ah<AH_number>'
N_AH_files = 2


# **IMPORTANT**
# Each path in AH_dirs is searched for the file containing the AH data for a
# given iteration via os.walk . In order to make life easier for os.walk, create
# a directory containing as many subdirectories as there are simulation
# restarts; each subdirectory should be a symlink to the actual directory
# containing all the AH files for the corresponding simulation restart.
AH_dirs = (
    ##"/lagoon/lennoggi/Snapshots/CBD_handoff_IGM_McLachlan_Spinning_aligned08_RadCool_OrbSep10M/AH_data",
    "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling_NewCooling/AH_data",
    "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_NonSpinning_large_TaperedCooling_NewCooling/AH_data"
)



# --------------
# Scale settings
# --------------

# Use a logarithmic scale?
logscale = (
    True,
    ##True
)

# Use a symmetric logarithmic scale? I.e., if a logarithmic scale is in use and
# data values extend down to zero, then a linear scale is used from zero to the
# desired minimum in the colorbar (see below)
symlogscale = (
    False,
    False
)

# If a linear color scale is in use (i.e. if 'logscale' is false), normalize it
# between 0 and 1?
linscale_norm = (
    True,
    True
)



# ----------
# Plot setup
# ----------

# Units
#   - 1 solar mass time     = 4.9257949707731345e-03 ms
#   - 1 solar mass distance = 1.4767161818921162 km
units = "arbitrary"  # "arbitrary", "geometric" or "SI"

# Names of the variables to be put close to the colorbar
varnames = (
    "$\\rho$",
    "$\\rho$"
)

# Titles for each subplot
titles = (
    "",
    ""
)

# Add colorbar(s)?
add_clb = (
    True,
    True
)

# Extent of the color scales (note that the actual scale may extend below
# colorbar_extents[i][0] if logscale[i] = True and symlogscale[i] = True)
clb_extents = (
    ##(1.e-08, 1.e-03),
    (1.e-08, 3.e-02),
    (1.e-12, 3.e-02)
)

# Type of colorbar extension outside its limits ("neither", "max", "min" or
# "both")
clb_extends = (
    "both",
    "both"
)

# Colormap
cmaps = (
    "plasma",
    "plasma"
)

# Title options
titlecolor       = "midnightblue"
titlepad         = 1.02
title_fontsize   = 35.
title_fontweight = "bold"
title_fontstyle  = "normal"
title_fontname   = "Ubuntu"

# Labels options
labelsize  = 25.
labelpad_x = 3.
labelpad_y = -5.
ticksize   = 20.

# Colorbar label options
clb_ticksize        = 20.
clblabel_pad        = 40. ##8.
clb_fraction        = 0.05
clblabel_fontsize   = 25.
clblabel_fontweight = "bold"
clblabel_fontstyle  = "normal"

# Iteration, time and orbits strings options
it_pos                 = (0.35, 0.015)
##time_pos               = (0.6,  0.015)
time_pos               = (0.45, 0.015)
orb_pos                = (0.12, 0.015)
it_time_orb_fontsize   = 20.
##it_time_orb_fontsize   = 25.
it_time_orb_fontweight = "bold"
it_time_orb_fontstyle  = "normal"



# -----------------------
# Dynamic zooming options
# -----------------------

# Zoom in/out as time goes?
zooms = (
    False,
    False
)

# If zooming in/out, set the actual plot extents at the final time here
actual_plot_extents_end = (
    (-40.,  40.,  -40.,  40.),
    (-200., 200., -200., 200.)
)


# Iterations at which zooming in/out should begin/end
first_its_zoom = (
    0,
    0
)

last_its_zoom = (
    1,
    1
)



# ------------------------------------------
# Dynamic grid plotting (pcolormesh) options
# ------------------------------------------

# Plot the grid using the 'edgecolor' option in np.pcolormesh?
# NOTE: this works best on uniform grids, as it shows the full
#       mesh and not just the refinement level boundaries
plot_grid = (
    False,
    False
)

# If plot_grid is true, do you want the grid to gradually fade in/out?
vary_grid_transparency = (
    False,
    False
)


# Iterations at which the change in grid transparency should begin/end
first_its_alpha_grid = (
    0,
    0
)

last_its_alpha_grid = (
    1,
    1
)


# Grid transparency values at the beginning/end
alpha_grid_init = (
    0.,
    0.
)

alpha_grid_end = (
    1.,
    1.
)



# -----------------------------------------------------------------
# Dynamic refinement level boundaries plotting (PostCactus) options
# -----------------------------------------------------------------

# Plot refinement level boundaries?
plot_reflevels = (
    False,
    False
)

# If plot_reflevels is true, do you want the refinement level boundaries to
# gradually fade in/out?
vary_reflevels_transparency = (
    False,
    False
)


# Iterations at which the change in refinement levels transparency should
# begin/end
first_its_alpha_reflevels = (
    0,
    0
)

last_its_alpha_reflevels = (
    1,
    1
)


# Refinement levels transparency at the beginning/end
alpha_reflevels_init = (
    1.,
    1.
)

alpha_reflevels_end = (
    0.,
    0.
)


# Range of refinement levels boundaries to be plotted
# NOTE: the minimum must be no smaller than 0, and you can set the maximum to
#       some high value to plot all reflevel boundaries
reflevel_ranges = (
    (0, 20),
    (0, 20)
)

# ***** End parameters *****





# =============
# Sanity checks
# =============

nplots = len(datadirs)

if skip_restarts is not None: assert len(skip_restarts) == nplots

assert len(gfs) == nplots
if gfs_stream is not None:
    assert len(gfs_stream) == nplots
    for gf_stream in gfs_stream:
        assert len(gf_stream) == 2

assert input_coords == "Cartesian" or input_coords == "Exponential fisheye"

assert len(planes) == nplots
for plane in planes:
    assert plane == "xy" or plane == "xz" or plane == "yz"

assert len(abs_vals) == nplots
for abs_val in abs_vals:
    assert abs_val or not abs_val

assert first_it    >= 0
assert last_it     >= first_it
assert out2D_every >= 0

assert orb_count or not orb_count
assert omega > 0.

assert len(compute_min_max) == nplots
for comp in compute_min_max:
    assert comp or not comp


assert len(plot_extents) == nplots
for plot_extent in plot_extents:
    assert len(plot_extent) == 4
    assert plot_extent[1] > plot_extent[0]
    assert plot_extent[3] > plot_extent[2]

assert actual_plot_extents is None or len(actual_plot_extents) == nplots

if actual_plot_extents is not None:
    for actual_plot_extent in actual_plot_extents:
        assert len(actual_plot_extent) == 4
        assert actual_plot_extent[1] > actual_plot_extent[0]
        assert actual_plot_extent[3] > actual_plot_extent[2]


assert nsubplots_x > 0
assert nsubplots_y > 0
assert nsubplots_x*nsubplots_y == nplots

assert len(figsize) == 2
assert dpi > 0


assert len(limit_resolution) == nplots
for limit in limit_resolution:
    assert limit or not limit

assert len(resolution) == nplots
for res in resolution:
    assert res > 0.


assert len(draw_AH) == nplots
for draw in draw_AH:
    assert draw or not draw

assert N_AH_files > 0
assert len(AH_dirs) == nplots


assert len(logscale) == nplots
for log in logscale:
    assert log or not log

assert len(symlogscale) == nplots
for symlog in symlogscale:
    assert symlog or not symlog

assert len(linscale_norm) == nplots
for lin in linscale_norm:
    assert lin or not lin


assert units == "arbitrary" or units == "geometric" or units == "SI"

assert len(varnames) == nplots
assert len(titles)   == nplots


assert len(add_clb) == nplots
for aclb in add_clb:
    assert aclb or not aclb

assert len(clb_extents) == nplots
for clb_extt in clb_extents:
    assert len(clb_extt) == 2
    assert clb_extt[1] > clb_extt[0]

assert len(clb_extends) == nplots
for clb_extd in clb_extends:
    assert clb_extd == "min" or clb_extd == "max" or clb_extd == "both"


assert len(cmaps) == nplots

assert title_fontsize    > 0.
assert labelsize         > 0.
assert ticksize          > 0.
assert clb_ticksize      > 0.
assert clb_fraction      > 0.
assert clblabel_fontsize > 0.

assert len(it_pos)   == 2
assert len(time_pos) == 2
assert len(orb_pos)  == 2
assert it_time_orb_fontsize > 0.


assert len(zooms) == nplots
for zm in zooms:
    assert zm or not zm

assert len(actual_plot_extents_end) == nplots
for actual_plot_extent_end in actual_plot_extents_end:
    assert len(actual_plot_extent_end) == 4
    assert actual_plot_extent_end[1] > actual_plot_extent_end[0]
    assert actual_plot_extent_end[3] > actual_plot_extent_end[2]

assert len(first_its_zoom) == nplots
for i in range(nplots):
    if zooms[i]:
        assert firsts_it_zoom[i] >= first_it

assert len(last_its_zoom) == nplots
for i in range(nplots):
    if zooms[i]:
        assert last_its_zoom[i] <= last_it


assert len(plot_grid) == nplots
for pg in plot_grid:
    assert pg or not pg

assert len(vary_grid_transparency) == nplots
for vgt in vary_grid_transparency:
    assert vgt or not vgt

assert len(first_its_alpha_grid) == nplots
for i in range(nplots):
    if plot_grid[i] and vary_grid_transparency[i]:
        assert first_its_alpha_grid[i] >= first_it

assert len(last_its_alpha_grid) == nplots
for i in range(nplots):
    if plot_grid[i] and vary_grid_transparency[i]:
        assert last_it_alpha_grid <= last_it

assert len(alpha_grid_init) == nplots
for agi in alpha_grid_init:
    assert agi >= 0.

assert len(alpha_grid_end) == nplots
for age in alpha_grid_end:
    assert age >= 0.


assert len(plot_reflevels) == nplots
for plot_rlvl in plot_reflevels:
    assert plot_rlvl or not plot_rlvl

assert len(vary_reflevels_transparency) == nplots
for vrt in vary_reflevels_transparency:
    assert vrt or not vrt

assert len(first_its_alpha_reflevels) == nplots
for i in range(nplots):
    if plot_reflevels[i] and vary_reflevels_transparency[i]:
        assert first_its_alpha_reflevels[i] >= first_it

assert len(last_its_alpha_reflevels) == nplots
for i in range(nplots):
    if plot_reflevels[i] and vary_reflevels_transparency[i]:
        assert last_its_alpha_reflevels[i] <= last_it

assert len(alpha_reflevels_init) == nplots
for ari in alpha_reflevels_init:
    assert ari >= 0.

assert len(alpha_reflevels_end) == nplots
for are in alpha_reflevels_end:
    assert are >= 0.

assert len(reflevel_ranges) == nplots
for rr in reflevel_ranges:
    assert len(rr) == 2
    assert rr[1] >= rr[0]

# ***** End sanity checks *****





# ===========
# Units setup
# ===========

if units == "arbitrary":
    conv_fac_time  = 1.
    conv_fac_space = 1.
    unit_time_str  = "$\mathbf{M}$"
    unit_space_str = "[$\mathbf{M}$]"

    # TODO: add other potentially useful conversion factors
    for gf in gfs:
        if gf == "rho" or gf == "rho_b":
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{-2}\\right]$"
        elif gf == "press" or gf == "P":
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{2}\\right]$"
        elif gf == "eps":
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}\\right]$"
        elif gf == "smallb2" or gf == "b2small" or gf == "bcom_sq":
            conv_fac_gf = 1.
            unit_gf_str = "$\,\left[\mathbf{M}^{-1}\\right]$"
        else:
            conv_fac_gf = 1.
            unit_gf_str = ""
            warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'") 


elif units == "geometric":
    conv_fac_time  = 1.
    conv_fac_space = 1.
    unit_time_str  = "$\mathbf{M_{\\odot}}$"
    unit_space_str = "[$\mathbf{M_{\\odot}}$]"

    # TODO: add other potentially useful conversion factors
    for gf in gfs:
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
    for gf in gfs:
        if gf == "rho" or gf == "rho_b":
            conv_fac_gf = Msun_to_kg_over_m3
            unit_gf_str = "$\,\left[\\frac{kg}{m^3}\\right]$"
        elif gf == "press" or gf == "P":
            conv_fac_gf = Msun_to_N_over_m2
            unit_gf_str = "$\,\left[\\frac{N}{m^2}\\right]$"
        elif gf == "smallb2" or gf == "b2small" or gf == "bcom_sq":
            conv_fac_gf = Msun_to_N_over_A2
            unit_gf_str = "$\,\left[T\\right]$"
        else:
            conv_fac_gf = 1.
            unit_gf_str = ""
            warnings.warn("No known conversion to " + units + " units for grid function '" + gf + "'") 


else: raise RuntimeError("Unrecognized units \"" + units + "\"")

# ***** End units setup *****





# ==========
# Plot setup
# ==========
figname = plotdir + "/"

simdirs   = []
read_data = []

AHfile_cols1 = np.empty([nplots], dtype = int)
AHfile_cols2 = np.empty([nplots], dtype = int)

xlabels = []
ylabels = []
norms   = []

there_are_iters_avail = np.empty([nplots], dtype = bool)
last_valid_it         = np.empty([nplots], dtype = int)
last_valid_g          = []

xsteps = np.empty([nplots], dtype = np.ndarray)
ysteps = np.empty([nplots], dtype = np.ndarray)

xmin_plot = np.empty([nplots], dtype = float)
xmax_plot = np.empty([nplots], dtype = float)
ymin_plot = np.empty([nplots], dtype = float)
ymax_plot = np.empty([nplots], dtype = float)

alpha_grid       = np.empty([nplots], dtype = float)
alpha_grid_steps = np.empty([nplots], dtype = float)

alpha_reflevels       = np.empty([nplots], dtype = float)
alpha_reflevels_steps = np.empty([nplots], dtype = float)


for n in range(nplots):
    figname   += gfs[n] + "_" + planes[n] + "_"
    simdirs.append(SimDir(datadirs[n]))

    if skip_restarts is not None:
        simdirs[n].dirs = [d for d in simdirs[n].dirs if not any(skip_restart in d for skip_restart in skip_restarts[n])]

    if planes[n] == "xy":
        read_data.append(simdirs[n].grid.hdf5.xy.read)
        AHfile_cols1[n] = 3
        AHfile_cols2[n] = 4

        if input_coords == "Cartesian":
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("y$\,$" + unit_space_str)
        elif input_coords == "Exponential fisheye":
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)

    elif planes[n] == "xz":
        read_data.append(simdirs[n].grid.hdf5.xz.read)
        AHfile_cols1[n] = 3
        AHfile_cols2[n] = 5

        if input_coords == "Cartesian":
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)
        elif input_coords == "Exponential fisheye":
            xlabels.append("x$\,$" + unit_space_str)
            ylabels.append("y$\,$" + unit_space_str)

    elif planes[n] == "yz":
        read_data.append(simdirs[n].grid.hdf5.yz.read)
        AHfile_cols1[n] = 4
        AHfile_cols2[n] = 5

        if input_coords == "Cartesian":
            xlabels.append("y$\,$" + unit_space_str)
            ylabels.append("z$\,$" + unit_space_str)
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME
        elif input_coords == "Exponential fisheye":
            xlabels.append("$\\theta$")
            ylabels.append("$\phi$")
        # FIXME FIXME FIXME FIXME FIXME FIXME FIXME

    else:
        raise RuntimeError("Unrecognized plane \"" + planes[n] + "\"")


    if logscale[n] == True:
        if symlogscale[n] == True:
            norms.append(colors.SymLogNorm(vmin      = clb_extents[n][0],
                                           vmax      = clb_extents[n][1],
                                           linthresh = clb_extents[n][0]))
        elif symlogscale[n] == False:
            norms.append(colors.LogNorm(vmin = clb_extents[n][0],
                                        vmax = clb_extents[n][1]))
        else:
            raise RuntimeError("Please set symlogscale[" + str(n) + "] to either 'True' or 'False'")

    elif logscale[n] == False:
        if linscale_norm[n] == True:
            norms.append(colors.Normalize(vmin = clb_extents[n][0],
                                          vmax = clb_extents[n][1]))
        elif linscale_norm[n] == False:
            norms.append(None)
        else:
            raise RuntimeError("Please set linscale_norm[" + str(n) + "] to either 'True' or 'False'")
    else:
        raise RuntimeError("Please set logscale[" + str(n) + "] to either 'True' or 'False'")


    there_are_iters_avail[n] = True
    last_valid_it[n]         = first_it
    last_valid_g.append(None)


    if zooms[n]:
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
        assert half_range_x_init > 0.
        assert half_range_y_init > 0.

        half_range_x_end = 0.5*(xmax_end - xmin_end)
        half_range_y_end = 0.5*(ymax_end - ymin_end)
        assert half_range_x_end > 0.
        assert half_range_y_end > 0.

        assert first_its_zoom[n] >= first_it
        assert  last_its_zoom[n] <=  last_it
        assert (last_its_zoom[n] - first_its_zoom[n]) % out2D_every == 0
        n_zoom = int((last_its_zoom[n] - first_its_zoom[n])/out2D_every) + 1

        # If the grid is expanding (half_range_x_end > half_range_x_init), then
        # xsteps and ysteps are monotonically increasing, otherwise they are
        # decreasing
        xsteps[n] = np.logspace(np.log(half_range_x_init), np.log(half_range_x_end),
                                num = n_zoom, base = np.e, endpoint = True)
        ysteps[n] = np.logspace(np.log(half_range_y_init), np.log(half_range_y_end),
                                num = n_zoom, base = np.e, endpoint = True)


    if (input_coords == "Cartesian" or
       (input_coords != "Cartesian" and actual_plot_extents[n] is None)):
        xmin_plot[n] = plot_extents[n][0]
        xmax_plot[n] = plot_extents[n][1]
        ymin_plot[n] = plot_extents[n][2]
        ymax_plot[n] = plot_extents[n][3]
    else:  # Non-Cartesian coords, actual_plot_extents[n] is not None
        xmin_plot[n] = actual_plot_extents[n][0]
        xmax_plot[n] = actual_plot_extents[n][1]
        ymin_plot[n] = actual_plot_extents[n][2]
        ymax_plot[n] = actual_plot_extents[n][3]

    if plot_grid[n]:
        alpha_grid[n] = alpha_grid_init[n]
        if vary_grid_transparency[n]:
            assert (last_its_alpha_grid[n] - first_its_alpha_grid[n]) % out2D_every == 0
            alpha_grid_n        = (last_its_alpha_grid[n] - first_its_alpha_grid[n])/out2D_every
            alpha_grid_range    = alpha_grid_end[n] - alpha_grid_init[n]
            alpha_grid_steps[n] = alpha_grid_range/alpha_grid_n

    if plot_reflevels[n]:
        alpha_reflevels[n] = alpha_reflevels_init[n]
        if vary_reflevels_transparency[n]:
            assert (last_its_alpha_reflevels[n] - first_its_alpha_reflevels[n]) % out2D_every == 0
            alpha_reflevels_n        = (last_its_alpha_reflevels[n] - first_its_alpha_reflevels[n])/out2D_every
            alpha_reflevels_range    = alpha_reflevels_end[n] - alpha_reflevels_init[n]
            alpha_reflevels_steps[n] = alpha_reflevels_range/alpha_reflevels_n

# ***** End plot setup *****





# ====
# Plot
# ====
nframe = int(first_it/out2D_every)

# Toy grid used to check whether an iteration is available for a given dataset
xmin_largest = plot_extents[:, 0].max()
xmax_largest = plot_extents[:, 1].max()
ymin_largest = plot_extents[:, 2].max()
ymax_largest = plot_extents[:, 3].max()
g_toy = gd.RegGeom([2, 2], [xmin_largest, ymin_largest],
                      x1 = [xmax_largest, ymax_largest])

# Array storing the times of each dataset. Used to figure out the largest time
# between two dataset in case any of them is not evolving anymore.
times = np.empty([nplots])

if orb_count: omega_over_2pi = 0.5*omega/np.pi


for it in range(first_it, last_it + 1, out2D_every):
    print("***** Iteration " + str(it) + " *****\n")

    if nplots > 1:  # Can't have 'plt.subplots(1, 1)'
        fig, axes = plt.subplots(nsubplots_y, nsubplots_x, figsize = figsize, dpi = dpi)
    else:  # Single subplot -> No arguments
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

    fig.set_tight_layout(True)

    for n in range(nplots):
        # The axis object returned by plt.subplots() when there's only one
        # subplot is not subscriptable
        if nplots > 1: ax = axes[n]

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
        patch_toy = read_data[n](gfs[n], it,
                                 geom           = g_toy,
                                 adjust_spacing = True,
                                 order          = 0,
                                 outside_val    = 0.,
                                 level_fill     = False)


        # If iteration 'it' is not available for this dataset, then flag it. If
        # all the other datasets are flagged already, then it means there's
        # nothing else to be plotted: break. Otherwise, reset to the last
        # non-flagged iteration for this dataset.
        if patch_toy.time is None:
            warnings.warn("Setting dataset " + str(n) + " to the last valid iteration")
            there_are_iters_avail[n] = False
            no_iters_avail_count     = 0

            for avail in there_are_iters_avail:
               if not avail: no_iters_avail_count += 1

            assert no_iters_avail_count <= nplots

            if no_iters_avail_count == nplots:
                raise StopIteration  # Dirty hack to break a nested loop

            else: 
                patch_plot = read_data[n](gfs[n], last_valid_it[n],
                                          geom           = last_valid_g[n],
                                          adjust_spacing = True,
                                          order          = 0,
                                          outside_val    = 0.,
                                          level_fill     = False)
                if gfs_stream is not None:
                    patch_stream1 = read_data[n](gfs_stream[n][0], last_valid_it[n],
                                                 geom           = last_valid_g[n],
                                                 adjust_spacing = True,
                                                 order          = 0,
                                                 outside_val    = 0.,
                                                 level_fill     = False)
                    patch_stream2 = read_data[n](gfs_stream[n][1], last_valid_it[n],
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
            patches = read_data[n](gfs[n], it,
                                   geom           = None,
                                   adjust_spacing = True,
                                   order          = 0,
                                   outside_val    = 0.,
                                   level_fill     = False)

            if limit_resolution[n]:
                deltax_min = resolution[n]
                deltay_min = resolution[n]
                print("Dataset " + str(n) + ": grid spacing limited to ("
                      + str(deltax_min) + ", " + str(deltay_min) + ")")
            else:
                for i in range(len(patches)):
                    geom   = patches[i].geom()
                    deltas = geom.dx()
                    dx     = deltas[0]
                    dy     = deltas[1]

                    if i == 0:
                        deltax_min = dx
                        deltay_min = dy
                    else:
                        if dx < deltax_min: deltax_min = dx
                        if dy < deltay_min: deltay_min = dy

                print("Dataset " + str(n) + ": finest grid spacing is ("
                      + str(deltax_min) + ", " + str(deltay_min) + ")")


            xmin  = plot_extents[n][0]
            xmax  = plot_extents[n][1]
            ymin  = plot_extents[n][2]
            ymax  = plot_extents[n][3]

            Nx = int((xmax - xmin)/deltax_min)
            Ny = int((ymax - ymin)/deltay_min)

            g = gd.RegGeom([Nx, Ny], [xmin, ymin], x1 = [xmax, ymax])

            # Build the field and stream patches to be plotted, which are
            # resampled to a uniform grid (with the finest or custom grid
            # spacing)
            patch_plot = read_data[n](gfs[n], it,
                                      geom           = g,
                                      adjust_spacing = True,
                                      order          = 0,
                                      outside_val    = 0.,
                                      level_fill     = False)
            if gfs_stream is not None:
                patch_stream1 = read_data[n](gfs_stream[n][0], it,
                                             geom           = g,
                                             adjust_spacing = True,
                                             order          = 0,
                                             outside_val    = 0.,
                                             level_fill     = False)
                patch_stream2 = read_data[n](gfs_stream[n][1], it,
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
        #   PostCactus wouldn't snap to the finest available resolution, leaving
        #   the shape of patch_plot untouched.
        if abs_vals[n]: plot_data = np.absolute(patch_plot.data*conv_fac_gf)
        else:           plot_data = patch_plot.data*conv_fac_gf

        Nx_new = plot_data.shape[0]
        Ny_new = plot_data.shape[1]

        if Nx_new != Nx or Ny_new != Ny:
            warnings.warn("Dataset " + str(n) + ": grid reshaped from (" + str(Nx) + ", " + str(Ny) + ") to (" + str(Nx_new) + ", " + str(Ny_new) + ")")


        # Build the mesh for pcolormesh and transform to Cartesian coordinates
        # if needed
        # NOTE: there are Nx cells => Nx+1 edges (same for Ny)
        if input_coords == "Cartesian":
            x       = np.linspace(xmin, xmax, Nx_new)
            y       = np.linspace(ymin, ymax, Ny_new)
            mx, my  = np.meshgrid(x, y)
        elif input_coords == "Exponential fisheye":
            logr        = np.linspace(xmin, xmax, Nx_new)
            phi         = np.linspace(ymin, ymax, Ny_new)
            mlogr, mphi = np.meshgrid(logr, phi)
            mx          = np.exp(mlogr)*np.cos(mphi)
            my          = np.exp(mlogr)*np.sin(mphi)
        else:
            raise RuntimeError("Invalid input coordinates")


        # Build the image to plot
        # NOTE: since mxcoords, mycoords and np.transpose(plot_data) all have
        #   the same shape, shading = "auto" should produce shading = "nearest",
        #   meaning that data are cell-centered. On the other hand, if mxcoords
        #   and mycoords had one point more than np.transpose(plot_data) in each
        #   direction, than the data would be placed on cell vertices and
        #   shading = "auto" should produce shading = "flat".
        if plot_grid[n]:
            im = ax.pcolormesh(mx, my, np.transpose(plot_data),
                               shading = "auto", cmap = cmaps[n], norm = norms[n],
                               edgecolor = (0., 0., 0., alpha_grid[n]),
                               linewidth = 0.01)
        else:
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            im = ax.pcolormesh(mx, my, np.transpose(plot_data),
            ##im = ax.pcolormesh(mx, my, 4.*np.pi*np.transpose(plot_data),  # XXX: useful e.g. when comparing IllinoisGRMHD::smallb2 and GRHydro::bcom_sq
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
                               shading = "auto", cmap = cmaps[n], norm = norms[n])
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            ##if n == 0:
            ##    im = ax.pcolormesh(mx, my, np.transpose(plot_data),
            ##                       shading = "auto", cmap = cmaps[n], norm = norms[n])
            ##else:
            ##    im = ax.pcolormesh(my, mx, np.transpose(plot_data),
            ##                       shading = "auto", cmap = cmaps[n], norm = norms[n])
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX
            # XXX XXX XXX XXX XXX XXX

        # Build the stream plot if necessary
        if gfs_stream is not None:
            ##viz.plot_vectors((patch_stream1, patch_stream2))
            ax.streamplot(mx, my, np.transpose(patch_stream1.data), np.transpose(patch_stream2.data),
                          density = streamline_density, linewidth = 0.5,
                          color = "black", arrowsize = 1., arrowstyle = "-|>")
            ##ax.quiver(mx, my, np.transpose(patch_stream1.data), np.transpose(patch_stream2.data))

        if add_clb[n]:
            clb = fig.colorbar(im, ax = ax, extend = clb_extends[n],
                               fraction = clb_fraction)
            clb.ax.set_title(varnames[n] + unit_gf_str, pad = clblabel_pad,
                             fontsize   = clblabel_fontsize,
                             fontweight = clblabel_fontweight,
                             fontstyle  = clblabel_fontstyle)
            clb.ax.tick_params(labelsize = clb_ticksize)


        # Plot apparent horizons by finding the convex hull of the projection of
        # the points defining it (as found by AHFinderDirect) in the desired plane
        if draw_AH[n]:
            print("Dataset " + str(n) + ": trying to draw apparent horizon(s)...")

            # In case there are multiple files available for the same apparent
            # horizon, the following overwrites the AH that many times
            for r in range(1, N_AH_files + 1):
                AH_file = ("h.t" + str(it) + ".ah" + str(r) + ".gp"
                           if there_are_iters_avail[n]
                           else
                           "h.t" + str(int(last_valid_it[n])) + ".ah" + str(r) + ".gp")

                for root, dirs, files in os.walk(str(AH_dirs[n]), followlinks = True):  # Search symlinks too
                    if AH_file in files:
                        AH_data = np.loadtxt(root + "/" + AH_file)
                        hull    = ConvexHull(AH_data[:, [AHfile_cols1[n], AHfile_cols2[n]]])
                        xhull   = AH_data[:, AHfile_cols1[n]][hull.vertices]
                        yhull   = AH_data[:, AHfile_cols2[n]][hull.vertices]

                        ax.fill(xhull*conv_fac_space, yhull*conv_fac_space,
                                linewidth = 1., facecolor = "None", edgecolor = "black")

                        print("Dataset " + str(n) + ": apparent horizon " + str(r) + " drawn from AH file " + str(r))
                    else:
                        warnings.warn("Dataset " + str(n) + ": no file found for AH " + str(r))
            print("")


        """
        if compute_min_max[n]:
            if abs_vals[n]:
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


        if (zooms[n] and it >= first_its_zoom[n]
                     and it <   last_its_zoom[n]):
            assert (it - first_its_zoom[n]) % out2D_every == 0
            snapshot_index = int((it - first_its_zoom[n])/out2D_every)
            assert snapshot_index >= 0 and snapshot_index < xsteps[n].shape[0] and snapshot_index < ysteps[n].shape[0]

            dlog_x = xsteps[n][snapshot_index + 1] - xsteps[n][snapshot_index]
            dlog_y = ysteps[n][snapshot_index + 1] - ysteps[n][snapshot_index]

            # If the grid is expanding (half_range_x_end > half_range_x_init),
            # then xsteps and ysteps are monotonically increasing and so
            # dlog_x > 0; otherwise dlog_x < 0 (same for dlog_y)
            xmin_plot[n] -= dlog_x
            xmax_plot[n] += dlog_x
            ymin_plot[n] -= dlog_y
            ymax_plot[n] += dlog_y


        if (plot_grid[n] and vary_grid_transparency[n] and
            it >= first_its_alpha_grid[n] and
            it <   last_its_alpha_grid[n]):
            alpha_grid[n] += alpha_grid_steps[n]


        if plot_reflevels[n]:
            patch_contour = read_data[n](gfs[n], it,
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

            if (vary_reflevels_transparency[n] and
                it >= first_its_alpha_reflevels[n] and
                it <   last_its_alpha_reflevels[n]):
                alpha_reflevels[n] += alpha_reflevels_steps[n]


        times[n] = patch_plot.time

        if there_are_iters_avail[n]:
            last_valid_it[n] = it
            last_valid_g[n]  = g

        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        if n == 0:
            ax.add_artist(plt.Circle((0., 0.), 15., fill = False, linestyle = "--", linewidth = 1., color = "black"))
            ax.add_artist(plt.Circle((0., 0.), 20., fill = False, linestyle = "-",  linewidth = 1., color = "black"))
            ax.add_artist(plt.Circle((0., 0.), 30., fill = False, linestyle = "-",  linewidth = 1., color = "black"))
            ax.add_artist(plt.Circle((0., 0.), 40., fill = False, linestyle = "-",  linewidth = 1., color = "black"))
        ##elif n ==1:
        ##    ax.axvline(x = -15., linestyle = "--", linewidth = 1., color = "black")
        ##    ax.axvline(x =  15., linestyle = "--", linewidth = 1., color = "black")
        ##    ax.axvline(x = -20., linestyle = "-",  linewidth = 1., color = "black")
        ##    ax.axvline(x =  20., linestyle = "-",  linewidth = 1., color = "black")
        ##    ax.axvline(x = -30., linestyle = "-",  linewidth = 1., color = "black")
        ##    ax.axvline(x =  30., linestyle = "-",  linewidth = 1., color = "black")
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX



    """
    fig.text(it_pos[0], it_pos[1],
             "It = " + str(it),
             color      = "red",
             fontsize   = it_time_orb_fontsize,
             fontweight = it_time_orb_fontweight,
             fontstyle  = it_time_orb_fontstyle)
    """

    time = np.max(times)
    t    = (t0 + time)*conv_fac_time

    fig.text(time_pos[0], time_pos[1],
             "t = " + str("{:.2e}".format(t)) + unit_time_str,
             color      = "red",
             fontsize   = it_time_orb_fontsize,
             fontweight = it_time_orb_fontweight,
             fontstyle  = it_time_orb_fontstyle)

    if orb_count:
        fig.text(orb_pos[0], orb_pos[1],
                 "#orb = " + str(int(np.floor(omega_over_2pi*t))),
                 color      = "red",
                 fontsize   = it_time_orb_fontsize,
                 fontweight = it_time_orb_fontweight,
                 fontstyle  = it_time_orb_fontstyle)


    plt.savefig(figname + str("{:0>4d}".format(nframe)) + fig_ext)
    plt.close()
    nframe += 1
    print("Done\n\n\n")
