import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from postcactus.simdir import SimDir
from postcactus import grid_data as gd


######################### USER-DEFINED PARAMETERS ##############################

data_dir = "/lagoon/bbhdisk/CBD_SphericalNR/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP"
plot_dir = "/lagoon/lennoggi/Snapshots/CBD_493_140_280_SerialFFTfilter_64nodes_7OMP/Zoom"
fig_ext  = ".png"

gf           = "rho"
plane        = "xz"
input_coords = "Exponential fisheye"

patch_extent     = np.array([np.log(15.), np.log(2000.), 0., 2.*np.pi])
plot_extent_init = np.array([-300., 300., -300., 300.])
plot_extent_end  = np.array([-20., 20., -20., 20.])

n_snapshots = 100

it = 798800

figsize = [12., 10.]
dpi     = 200

vmin       = 1.e-08
vmax       = 1.5e-02
clb_extend = "max"

logscale      = True
symlogscale   = False
linscale_norm = True

cmap = "plasma"

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


# Load the data
sd = SimDir(data_dir)

# Unpack the grid parameters
patch_xmin = patch_extent[0]
patch_xmax = patch_extent[1]
patch_ymin = patch_extent[2]
patch_ymax = patch_extent[3]

# Build the desired geometry. The number of points in the grid, now just 2*2,
# be reset by PostCactus by snapping to the finest available iteration.
# FIXME: manually compute nx and ny
g = gd.RegGeom([200, 200], [patch_xmin, patch_ymin],
                      x1 = [patch_xmax, patch_ymax])


# Select the plane
if (plane == "xy"):
    read_data = sd.grid.xy.read
elif (plane == "xz"):
    read_data = sd.grid.xz.read
elif (plane == "yz"):
    read_data = sd.grid.yz.read
else:
    raise RuntimeError("Invalid plane \"" + plane + "\"")


# Build the patch
patch = read_data(gf, it, geom = g,
                  adjust_spacing = True, order = 0,
                  outside_val = 0., level_fill = False)


# Generate the mesh
data = patch.data

nx = data.shape[0]
ny = data.shape[1]

if (input_coords == "Cartesian"):
    x = np.linspace(patch_xmin, patch_xmax, nx)
    y = np.linspace(patch_ymin, patch_ymax, ny)
    mx, my = np.meshgrid(x, y)

elif (input_coords == "Exponential fisheye"):
    logr        = np.linspace(patch_xmin, patch_xmax, nx)
    phi         = np.linspace(patch_ymin, patch_ymax, ny)
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
fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
fig.set_tight_layout(True)

for n in range(n_snapshots):
    im = ax.pcolormesh(mx, my, np.transpose(data), shading = "auto",
                           cmap = cmap, norm = norm)
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)

    ax.set_aspect(1.)
    plt.savefig(plot_dir + "/" + gf + "_" + str("{:0>4d}".format(n)) + fig_ext)
    ax.cla()

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
