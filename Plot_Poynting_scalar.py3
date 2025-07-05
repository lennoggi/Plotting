import numpy as np
from postcactus.simdir import SimDir
from postcactus import grid_data as gd
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from scipy.spatial import ConvexHull
import os
import warnings

from matplotlib import rc
from matplotlib import rcParams
##rc("mathtext", usetex = True)
##rc("font",**{"family":"sans-serif","sans-serif":["Avant Garde"]})
##rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
rcParams["mathtext.fontset"] = "dejavuserif"


# ===== Parameters =====

simdir  = "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_ANALYSIS/output-0009/HDF5_2D"
plotdir = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_ANALYSIS"

it          = 999424 ##4284416 ##3022848 ##3753984 ##999424
out2D_every = 2048 ##1024 ##400
t0          = 99189.9 ##0.

plane       = "xz"
plot_extent = (-45., 45., 0., 90.)

limit_resolution = True
resolution       = 0.125

logscale   = True
clb_extent = (1.e-08, 1.e-03)
cmap       = "jet"

draw_AH   = True
N_AHfiles = 2
AH_dir    = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_ANALYSIS/AH_data"

Bstream = True

figsize = (11., 10.)
dpi     = 200
fig_ext = "jpg" ##"png"

# ======================


# Set up the data reader
sd = SimDir(simdir)

if plane == "xy":
    read_data   = sd.grid.hdf5.xy.read
    x1label     = "x$\,\left[\mathbf{M}\\right]$"
    x2label     = "y$\,\left[\mathbf{M}\\right]$"
    AHfile_col1 = 3
    AHfile_col2 = 4
elif plane == "xz":
    read_data   = sd.grid.hdf5.xz.read
    x1label     = "x$\,\left[\mathbf{M}\\right]$"
    x2label     = "z$\,\left[\mathbf{M}\\right]$"
    AHfile_col1 = 3
    AHfile_col2 = 5
elif plane == "yz":
    read_data   = sd.grid.hdf5.yz.read
    x1label     = "y$\,\left[\mathbf{M}\\right]$"
    x2label     = "z$\,\left[\mathbf{M}\\right]$"
    AHfile_col1 = 4
    AHfile_col2 = 5
else:
    raise RuntimeError(f"Invalid plane '{plane}'")


# Set the resolution
if limit_resolution:
    dx1min = resolution
    dx2min = resolution
    print(f"Resolution limited to {dx1min}")

else:
    patches = read_data("gxx", it, geom = None, adjust_spacing = True,
                    order = 0, outside_val = 0., level_fill = False)

    for i in range(len(patches)):
        geom   = patches[i].geom()
        deltas = geom.dx()
        dx1    = deltas[0]
        dx2    = deltas[1]

        if i == 0:
            dx1min = dx1
            dx2min = dx2
        else:
            if dx1 < dx1min: dx1min = dx1
            if dx2 < dx2min: dx2min = dx2

    print(f"Resolution set to {dx1min} (finest available)")

x1min, x1max, x2min, x2max = plot_extent
Nx1 = int((x1max - x1min)/dx1min)
Nx2 = int((x2max - x2min)/dx2min)


# Build the geometry and load the data
g = gd.RegGeom([Nx1, Nx2], [x1min, x2min], x1 = [x1max, x2max])

gxx = read_data("gxx", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False)

t = t0 + gxx.time

gxx = gxx.data
gxy = read_data("gxy", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
gxz = read_data("gxz", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
gyy = read_data("gyy", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
gyz = read_data("gyz", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
gzz = read_data("gzz", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data

vx = read_data("vel[0]", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
vy = read_data("vel[1]", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data
vz = read_data("vel[2]", it, geom = g, adjust_spacing = True,
                order = 0, outside_val = 0., level_fill = False).data

Bx = read_data("Bx", it, geom = g, adjust_spacing = True,
               order = 0, outside_val = 0., level_fill = False).data
By = read_data("By", it, geom = g, adjust_spacing = True,
               order = 0, outside_val = 0., level_fill = False).data
Bz = read_data("Bz", it, geom = g, adjust_spacing = True,
               order = 0, outside_val = 0., level_fill = False).data


# Build the Poynting scalar (magnitude of the Poynting vector)
Bxlow = gxx*Bx + gxy*By + gxz*Bz
Bylow = gxy*Bx + gyy*By + gyz*Bz
Bzlow = gxz*Bx + gyz*By + gzz*Bz

Bsq   = Bx*Bxlow + By*Bylow + Bz*Bzlow
vdotB = vx*Bxlow + vy*Bylow + vz*Bzlow

Sx = Bsq*vx - vdotB*vx
Sy = Bsq*vy - vdotB*vy
Sz = Bsq*vz - vdotB*vz

Sxlow = gxx*Sx + gxy*Sy + gxz*Sz
Sylow = gxy*Sx + gyy*Sy + gyz*Sz
Szlow = gxz*Sx + gyz*Sy + gzz*Sz

Smag = np.sqrt(Sx*Sxlow + Sy*Sylow + Sz*Szlow)


# Build the mesh for the plot    
Nx1_new = Smag.shape[0]
Nx2_new = Smag.shape[1]

if Nx1_new != Nx1 or Nx2_new != Nx2:
    warnings.warn(f"Grid reshaped from ({Nx1}, {Nx2}) to ({Nx1_new}, {Nx2_new})")

x1       = np.linspace(x1min, x1max, Nx1_new)
x2       = np.linspace(x2min, x2max, Nx2_new)
mx1, mx2 = np.meshgrid(x1, x2)



# ------------------------
# Plot the Poynting scalar
# ------------------------
fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
fig.set_tight_layout(True)
ax.set_box_aspect(1)  # Square snapshots

ax.set_xlabel(x1label, fontsize = 25.) ##, labelpad = 3.)
ax.set_ylabel(x2label, fontsize = 25., labelpad = 20.)

ax.tick_params(labelsize = 20.)
ax.set_xlim(x1min, x1max)
ax.set_ylim(x2min, x2max)

##if logscale: norm = colors.LogNorm(  vmin = clb_extent[0], vmax = clb_extent[1])
if logscale: norm = colors.SymLogNorm(vmin = clb_extent[0], vmax = clb_extent[1], linthresh = clb_extent[0])
else:        norm = colors.Normalize( vmin = clb_extent[0], vmax = clb_extent[1])

im = ax.pcolormesh(mx1, mx2, np.transpose(Smag),
                   shading = "auto", cmap = cmap, norm = norm)

clb = fig.colorbar(im, ax = ax, extend = "both", fraction = 0.05)
clb.ax.set_title("$S\,\left[\mathbf{M}^{-1}\\right]$", pad = 40., fontsize = 25.) ##, pad = 40.) ##, fontweight = "bold", fontstyle  = "normal")
clb.ax.tick_params(labelsize = 20.)


# Plot the magnetic field lines if requested
if Bstream:
    if plane == "xy":
        B1stream = Bx
        B2stream = By
    elif plane == "xz":
        B1stream = Bx
        B2stream = Bz
    elif plane == "yz":
        B1stream = By
        B2stream = Bz
    else:
        raise RuntimeError(f"Invalid plane '{plane}'")

    ax.streamplot(mx1, mx2, np.transpose(B1stream), np.transpose(B2stream),
                  density = 3., linewidth = 1., color = "black", arrowsize = 1.5, arrowstyle = "-|>")


# Plot apparent horizons by finding the convex hull of the projection of
# the points defining it (as found by AHFinderDirect) in the desired plane
if draw_AH:
    # In case there are multiple files available for the same apparent
    # horizon, the following overwrites the AH that many times
    for r in range(1, N_AHfiles + 1):
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        AHfile = f"h.t{it}.ah{r}.gp"
        ##AHfile = f"h.t2138112.ah{r}.gp"
        ##AHfile = f"h.t4087808.ah{r}.gp"
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX
        # XXX XXX XXX XXX XXX XXX

        for root, dirs, files in os.walk(AH_dir, followlinks = True):  # Search symlinks too
            if AHfile in files:
                AHdata = np.loadtxt(root + "/" + AHfile)
                hull   = ConvexHull(AHdata[:, [AHfile_col1, AHfile_col2]])
                x1hull = AHdata[:, AHfile_col1][hull.vertices]
                x2hull = AHdata[:, AHfile_col2][hull.vertices]

                ax.fill(x1hull, x2hull, linewidth = 2., facecolor = "None", edgecolor = "black", zorder = 3)
                print(f"Apparent horizon {r} drawn from AH file '{AHfile}'")
            else:
                warnings.warn(f"AH file '{AHfile}' not found")


# Finish up with the plot
fig.text(0.6, 0.02, f"t = {t}" + "$\,\mathbf{M}$", fontsize = 20., fontweight = "bold", color = "red")
figname = f"{plotdir}/Smag_{plane}_{int(it/out2D_every):04}.{fig_ext}"
plt.savefig(figname)
plt.close()
print(f"File '{figname}' generated successfully")
