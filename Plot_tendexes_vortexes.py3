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

simdir  = "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_LateInspiral/output-0003/HDF5_2D"
plotdir = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_LateInspiral"

it          = 3941376 ##3938304
out2D_every = 512 ##2048 ##1024 ##400
t0          = 99189.9 ##0.

plane       = "xy"
plot_extent = (-10., 10., -10., 10.)

limit_resolution = True
resolution       = 0.015625

logscale   = True
cmap       = "jet"

clb_extents = (
    (1.e-20, 1.e-15),  # Tendex eigenvalue 1
    (1.e-20, 1.e-03),  # Tendex eigenvalue 2
    (1.e-04, 1.e+00),  # Tendex eigenvalue 3
    (1.e-20, 1.e-15),  # Vortex eigenvalue 1
    (1.e-15, 1.e-05),  # Vortex eigenvalue 2
    (1.e-05, 1.e+00),  # Vortex eigenvalue 3
)

draw_AH   = True
N_AHfiles = 2
AH_dir    = "/scratch3/07825/lennoggi/Movies/BBH_handoff_McLachlan_pp08_large_14rl_NewCooling_LateInspiral/AH_data"

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
    patches = read_data("rpsi0", it, geom = None, adjust_spacing = True,
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

rpsi0 = read_data("rpsi0", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False)

t = t0 + rpsi0.time

rpsi0 = rpsi0.data
rpsi1 = read_data("rpsi1", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
rpsi2 = read_data("rpsi2", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
rpsi3 = read_data("rpsi3", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
rpsi4 = read_data("rpsi4", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data

ipsi0 = read_data("ipsi0", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
ipsi1 = read_data("ipsi1", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
ipsi2 = read_data("ipsi2", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
ipsi3 = read_data("ipsi3", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data
ipsi4 = read_data("ipsi4", it, geom = g, adjust_spacing = True,
                  order = 0, outside_val = 0., level_fill = False).data


# Build the tendex ("electric") and vortex ("magnetic") tensors
rpsi0_p_rpsi4      = rpsi0 + rpsi4
rpsi0_p_rpsi4_half = 0.5*rpsi0_p_rpsi4

ipsi0_p_ipsi4      = ipsi0 + ipsi4
ipsi0_p_ipsi4_half = 0.5*ipsi0_p_ipsi4

E11 =  2.*rpsi2
E12 =  rpsi3 - rpsi1
E13 = -ipsi1 - ipsi3
E22 =  rpsi0_p_rpsi4_half - rpsi2
E23 =  ipsi0_p_ipsi4_half
E33 = -rpsi0_p_rpsi4_half - rpsi2

B11 =  2.*ipsi2
B12 =  ipsi3 - ipsi1
B13 =  rpsi1 + rpsi3
B22 =  ipsi0_p_ipsi4_half - ipsi2
B23 =  0.5*(rpsi4 - rpsi0)
B33 = -ipsi0_p_ipsi4_half - ipsi2


# Diagonalize E and B to get the tendex and vortex fields. At each point in
# spacetime (or rather, for a fixed time, at each point on the 3D spatial
# spacetime slice), you have 3 eigenvectors for E (B) representing the 3
# principal tidal (frame-dragging) directions with the associated 3 eigenvalues:
# positive eignevalue means a test rod (gyroscope) stretches (differentially
# rotates in one direction), while positive eigenvalue means a test rod
# (gyroscope) compresses (differentially rotates in the other direction). So,
# there are 3 independent vector fields associated with E and another 3
# independent vector fields associated with B.
E = np.array([[E11, E12, E13],
              [E12, E22, E23],
              [E13, E23, E33]])
B = np.array([[B11, B12, B13],
              [B12, B22, B23],
              [B13, B23, B33]])

# Currently, E and B are such that the first two axes (0 and 1) represent the
# location indices in the slice, while the last two axes (2 and 3) are the
# matrix indices. Reshape E and B to swap these pair of indices, so that the
# resulting "matrix fields" (tensor fields) can be meaningfully passed to
# np.linalg.eigh.
E = E.transpose(2, 3, 0, 1)
B = B.transpose(2, 3, 0, 1)

tendex_eigenvalues, tendex_fields = np.linalg.eigh(E)
vortex_eigenvalues, vortex_fields = np.linalg.eigh(B)

tendex_eigenvalue_1 = tendex_eigenvalues[:, :, 0]  # Single number at each spacetime point
tendex_eigenvalue_2 = tendex_eigenvalues[:, :, 1]
tendex_eigenvalue_3 = tendex_eigenvalues[:, :, 2]

vortex_eigenvalue_1 = vortex_eigenvalues[:, :, 0]
vortex_eigenvalue_2 = vortex_eigenvalues[:, :, 1]
vortex_eigenvalue_3 = vortex_eigenvalues[:, :, 2]

ftendex1 = tendex_fields[:, :, 0, :]  # Vector with 3 elements at each spacetime point
ftendex2 = tendex_fields[:, :, 1, :]
ftendex3 = tendex_fields[:, :, 2, :]

fvortex1 = vortex_fields[:, :, 0, :]
fvortex2 = vortex_fields[:, :, 1, :]
fvortex3 = vortex_fields[:, :, 2, :]

# Build the mesh for the plot    
Nx1_new = ftendex1.shape[0]
Nx2_new = ftendex1.shape[1]

if Nx1_new != Nx1 or Nx2_new != Nx2:
    warnings.warn(f"Grid reshaped from ({Nx1}, {Nx2}) to ({Nx1_new}, {Nx2_new})")

x1       = np.linspace(x1min, x1max, Nx1_new)
x2       = np.linspace(x2min, x2max, Nx2_new)
mx1, mx2 = np.meshgrid(x1, x2)


# Plot the three tidal stretching and the three frame dragging magnitudes (i.e.,
# eigenvalues) along with the corresponding principal directions vector fields
# (i.e., eigenvectors)
eigentuples = (
   (tendex_eigenvalue_1, ftendex1, clb_extents[0], "T1", "$T_1$"),
   (tendex_eigenvalue_2, ftendex2, clb_extents[1], "T2", "$T_2$"),
   (tendex_eigenvalue_3, ftendex3, clb_extents[2], "T3", "$T_3$"),
   (vortex_eigenvalue_1, fvortex1, clb_extents[3], "V1", "$V_1$"),
   (vortex_eigenvalue_2, fvortex2, clb_extents[4], "V2", "$V_2$"),
   (vortex_eigenvalue_3, fvortex3, clb_extents[5], "V3", "$V_3$")
)

for eigenvalue, eigenvector, clb_extent, eigenname, eigenname_clb in eigentuples:
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
    
    im = ax.pcolormesh(mx1, mx2, np.transpose(eigenvalue),
                       shading = "auto", cmap = cmap, norm = norm)
    
    clb = fig.colorbar(im, ax = ax, extend = "both", fraction = 0.05)
    clb.ax.set_title(eigenname_clb, pad = 40., fontsize = 25.) ##, pad = 40.) ##, fontweight = "bold", fontstyle  = "normal")
    clb.ax.tick_params(labelsize = 20.)

    if plane == "xy":
        f1stream = eigenvector[:, :, 0]
        f2stream = eigenvector[:, :, 1]
    elif plane == "xz":
        f1stream = eigenvector[:, :, 0]
        f2stream = eigenvector[:, :, 2]
    elif plane == "yz":
        f1stream = eigenvector[:, :, 1]
        f2stream = eigenvector[:, :, 2]
    else:
        raise RuntimeError(f"Invalid plane '{plane}'")

    ax.streamplot(mx1, mx2, np.transpose(f1stream), np.transpose(f2stream),
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
    figname = f"{plotdir}/{eigenname}_{plane}_{int(it/out2D_every):04}.{fig_ext}"
    plt.savefig(figname)
    plt.close()
    print(f"File '{figname}' generated successfully")
