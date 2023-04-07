import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from postcactus.simdir import SimDir
from postcactus import grid_data as gd
import warnings

from matplotlib import rcParams
rcParams["mathtext.fontset"] = "dejavuserif"

######################### USER-DEFINED PARAMETERS ##############################

data_dir = "/lagoon/bbhdisk/CBD_SphericalNR/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP"
plot_dir = "/lagoon/lennoggi/Snapshots/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP/Zoom"
fig_ext  = ".png"

gf           = "rho"
plane        = "xz"
it           = 798800
input_coords = "Exponential fisheye"
n_snapshots  = 150

patch_extent     = np.array([np.log(15.1), np.log(2000.), 0., 2.*np.pi])
plot_extent_init = np.array([-300., 300., -300., 300.])
plot_extent_end  = np.array([-40., 40., -40., 40.])

gf_name = "$\\rho$"
units   = "arbitrary"

logscale      = True
symlogscale   = False
linscale_norm = True

plot_grid  = False
grid_alpha = 0.5

cmap = "plasma"

figsize = [12., 10.]
dpi     = 200

vmin = 1.e-08
vmax = 1.5e-02

add_colorbar        = True
clb_extend          = "max"
clb_ticksize        = 20.
clblabel_pad        = 8.
clb_fraction        = 0.05
clblabel_fontsize   = 25.
clblabel_fontweight = "bold"
clblabel_fontstyle  = "normal"

labelsize  = 25.
labelpad_x = 3.
labelpad_y = -5.
ticksize   = 20.

it_pos             = np.array([0.15, 0.015])
time_pos           = np.array([0.6, 0.015])
it_time_fontsize   = 25.
it_time_fontweight = "bold"
it_time_fontstyle  = "normal"

################################################################################


# Check some parameters
assert(patch_extent[1] > patch_extent[0])
assert(patch_extent[3] > patch_extent[2])

assert(plot_extent_init[1] > plot_extent_init[0])
assert(plot_extent_init[3] > plot_extent_init[2])

assert(plot_extent_end[1] > plot_extent_end[0])
assert(plot_extent_end[3] > plot_extent_end[2])

assert(n_snapshots > 0)
assert(it          > 0)


# Set up the units
# ----------------
if (units == "arbitrary"):
    conv_fac_time  = 1.
    unit_time_str  = "$\mathbf{M}$"

    ##conv_fac_space = 1.  # Only needed when plotting AHs
    unit_space_str = "[$\mathbf{M}$]"

    # TODO: add other conversions
    if (gf == "rho" or gf == "rho_b"):
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

    ##conv_fac_space = 1.  # Only needed when plotting AHs
    unit_space_str = "[$\mathbf{M_{\\odot}}$]"

    # TODO: add other conversions
    if (gf == "rho" or gf == "rho_b"):
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{-2}\\right]$"
    elif (gf == "press" or gf == "P"):
        conv_fac_gf = 1.
        unit_gf_str = "$\,\left[\mathbf{M_{\\odot}}^{2}\\right]$"
    elif (gf_name == "eps"):
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

    ##conv_fac_space = 0.001*Msun_to_m  # From solar masses to kilometers    # Only needed when plotting AHs
    conv_fac_time  = 1000.*Msun_to_s  # From solar masses to milliseconds

    unit_time_str  = " $\mathbf{ms}$"
    unit_space_str = "[$\mathbf{km}$]"

    # TODO: add other conversion
    if (gf == "rho" or gf == "rho_b"):
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

else:
    raise RuntimeError("Unrecognized units \"" + units + "\"")


# Load the data
sd = SimDir(data_dir)


# Select the plane
if (plane == "xy"):
    read_data = sd.grid.xy.read
    if (input_coords == "Cartesian"):
       xlabel = "x$\,$" + unit_space_str
       ylabel = "y$\,$" + unit_space_str
    elif (input_coords == "Exponential fisheye"):
       xlabel = "x$\,$" + unit_space_str
       ylabel = "z$\,$" + unit_space_str

elif (plane == "xz"):
    read_data = sd.grid.xz.read
    if (input_coords == "Cartesian"):
       xlabel = "x$\,$" + unit_space_str
       ylabel = "z$\,$" + unit_space_str
    elif (input_coords == "Exponential fisheye"):
       xlabel = "x$\,$" + unit_space_str
       ylabel = "y$\,$" + unit_space_str

elif (plane == "yz"):
    read_data = sd.grid.yz.read
    if (input_coords == "Cartesian"):
       xlabel = "y$\,$" + unit_space_str
       ylabel = "z$\,$" + unit_space_str
    # FIXME FIXME FIXME FIXME FIXME FIXME FIXME
    elif (input_coords == "Exponential fisheye"):
       xlabel = "$\\theta$"
       ylabel = "$\\phi$"
    # FIXME FIXME FIXME FIXME FIXME FIXME FIXME

else:
    raise RuntimeError("Invalid plane \"" + plane + "\"")


# Build an object containing the grid hierarchy, i.e. a list of objects each
# containing information about the grid patch (size, resolution, time,
# iteration, refinement level, number of ghost cells, etc.) and the associated
# data 
patches = read_data(gf, it, geom = None,
                    adjust_spacing = True, order = 0,
                    outside_val = 0., level_fill = False)


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

print("Finest grid spacing: (" + str(deltax_min) + ", " + str(deltay_min) + ")")


# Set the geometry of the grid to be plotted
patch_xmin = patch_extent[0]
patch_xmax = patch_extent[1]
patch_ymin = patch_extent[2]
patch_ymax = patch_extent[3]

nx = int((patch_xmax - patch_xmin)/deltax_min)
ny = int((patch_ymax - patch_ymin)/deltay_min)

g = gd.RegGeom([nx, ny], [patch_xmin, patch_ymin],
                    x1 = [patch_xmax, patch_ymax])


# Build the patch for plotting
# NOTE: PostCactus may reshape the grid in order to match the finest resolution
patch = read_data(gf, it, geom = g,
                  adjust_spacing = True, order = 0,
                  outside_val = 0., level_fill = False)


# Generate the mesh
data = patch.data*conv_fac_gf

nx_new = data.shape[0]
ny_new = data.shape[1]

if (nx_new != nx or ny_new != ny):
    print("Grid reshaped from (" + str(nx) + ", " + str(ny) + ") to (" + str(nx_new) + ", " + str(ny_new) + ")")

if (input_coords == "Cartesian"):
    x = np.linspace(patch_xmin, patch_xmax, nx_new)
    y = np.linspace(patch_ymin, patch_ymax, ny_new)
    mx, my = np.meshgrid(x, y)

elif (input_coords == "Exponential fisheye"):
    logr        = np.linspace(patch_xmin, patch_xmax, nx_new)
    phi         = np.linspace(patch_ymin, patch_ymax, ny_new)
    mlogr, mphi = np.meshgrid(logr, phi)
    mx          = np.exp(mlogr)*np.cos(mphi)
    my          = np.exp(mlogr)*np.sin(mphi)
else:
    raise RuntimeError("Invalid input coordinates")
    


# Set up the normalization
if (logscale):
    if (symlogscale):
        norm = colors.SymLogNorm(vmin = vmin, vmax = vmax, linthresh = vmin)
    else:
        norm = colors.LogNorm(vmin = vmin, vmax = vmax)
elif (linnorm):
    norm = colors.Normalize(vmin = vmin, vmax = vmax)
else:
    norm = None


# Compute the plot size reduction to be applied at each timestep,
# logarithmically increasing/decreasing the zoom
xmin_init = plot_extent_init[0]
xmax_init = plot_extent_init[1]
ymin_init = plot_extent_init[2]
ymax_init = plot_extent_init[3]

xmin_end = plot_extent_end[0]
xmax_end = plot_extent_end[1]
ymin_end = plot_extent_end[2]
ymax_end = plot_extent_end[3]

half_range_x_init = 0.5*(xmax_init - xmin_init)
half_range_y_init = 0.5*(ymax_init - ymin_init)
assert(half_range_x_init > 0.)
assert(half_range_y_init > 0.)

half_range_x_end = 0.5*(xmax_end - xmin_end)
half_range_y_end = 0.5*(ymax_end - ymin_end)
assert(half_range_x_end > 0.)
assert(half_range_y_end > 0.)

# If the grid is expanding (half_range_x_end > half_range_x_init), then xsteps
# and ysteps are monotonically increasing, otherwise they are decreasing
xsteps = np.logspace(np.log(half_range_x_init), np.log(half_range_x_end),
                     num = n_snapshots, base = np.e, endpoint = True)
ysteps = np.logspace(np.log(half_range_y_init), np.log(half_range_y_end),
                     num = n_snapshots, base = np.e, endpoint = True)


# Initialize the plot limits
xmin_plot = xmin_init
xmax_plot = xmax_init
ymin_plot = ymin_init
ymax_plot = ymax_init


# Plot multiple snapshots with increasing/decreasing zoom
"""
fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

fig.set_tight_layout(True)

ax.set_aspect(1.)

ax.set_xlabel(xlabel, fontsize = labelsize, labelpad = labelpad_x)
ax.set_ylabel(ylabel, fontsize = labelsize, labelpad = labelpad_y)

ax.tick_params(labelsize = ticksize)
"""

for n in range(n_snapshots):
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)

    fig.set_tight_layout(True)
    ax.set_aspect(1.)

    ax.set_xlabel(xlabel, fontsize = labelsize, labelpad = labelpad_x)
    ax.set_ylabel(ylabel, fontsize = labelsize, labelpad = labelpad_y)

    ax.tick_params(labelsize = ticksize)

    if (plot_grid):
        im = ax.pcolormesh(mx, my, np.transpose(data),
                           shading = "auto", cmap = cmap, norm = norm,
                           ##alpha = grid_alpha,
                           edgecolor = "black", linewidth = 0.1)
    else:
        im = ax.pcolormesh(mx, my, np.transpose(data),
                           shading = "auto", cmap = cmap, norm = norm)
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)

    if (add_colorbar):
        clb = fig.colorbar(im, ax = ax, extend = clb_extend,
                           fraction = clb_fraction)
        clb.ax.set_title(gf_name + unit_gf_str, pad = clblabel_pad,
                         fontsize   = clblabel_fontsize,
                         fontweight = clblabel_fontweight,
                         fontstyle  = clblabel_fontstyle)
        clb.ax.tick_params(labelsize = clb_ticksize)

    # Time and iteration info
    it_str   = "It = " + str(it)
    time_str = "t = " + str("{:.2e}".format(patch.time*conv_fac_time))

    fig.text(it_pos[0], it_pos[1], it_str, color = "red",
             fontsize  = it_time_fontsize,  fontweight = it_time_fontweight,
             fontstyle = it_time_fontstyle)
    fig.text(time_pos[0], time_pos[1], time_str + unit_time_str, color = "red",
             fontsize  = it_time_fontsize,  fontweight = it_time_fontweight,
             fontstyle = it_time_fontstyle)

    plt.savefig(plot_dir + "/" + gf + "_" + str("{:0>4d}".format(n)) + fig_ext)
    ##ax.cla()
    ##fig.clf()

    if (n < n_snapshots - 1):
        # If the grid is expanding (half_range_x_end > half_range_x_init), then
        # then xsteps and ysteps are monotonically increasing and so dlog_x > 0,
        # otherwise dlog_x < 0 (same for dlog_y)
        dlog_x = xsteps[n + 1] - xsteps[n]
        dlog_y = ysteps[n + 1] - ysteps[n]

        xmin_plot -= dlog_x
        xmax_plot += dlog_x
        ymin_plot -= dlog_y
        ymax_plot += dlog_y

    print("Snapshot " + str(n) + " plotted successfully")

    plt.close(fig)
