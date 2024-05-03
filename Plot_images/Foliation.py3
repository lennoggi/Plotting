import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Hypersurface functions
def Sigma_t(x, y):
    return 0.2*np.sin(-0.2*x) - 0.2*np.sin(0.3*y) - 2.5
def Sigma_tdt(x, y):
    return -0.1*np.sin(0.2*x + 0.3*y) - 0.2*np.sin(0.2*y) + 3.

# Colors
mydarkblue = (0., 0.05, 0.5)
mylightblue  = (0.72, 0.8, 0.95)



# ==================
# Plot hypersurfaces
# ==================
##fig   = plt.figure()
fig   = plt.figure(dpi = 600)
axes  = fig.add_subplot(111, projection = "3d")
axes.set_zlim(bottom = -2., top = 2.)
axes.set_axis_off()

x    = np.arange(-10., 10., 0.1)
y    = np.arange(-10., 10., 0.1)
x, y = np.meshgrid(x, y)

# \Sigma_t surface
axes.plot_surface(x, y, Sigma_t(x, y), antialiased = False,
                  color = "lightskyblue", alpha = 0.6, shade = True)
axes.text2D(0.92, 0.28, "$\Sigma_t$",
            transform = axes.transAxes, fontsize = 25.)

# \Sigma_{t + \delta_t} surface
axes.plot_surface(x, y, Sigma_tdt(x, y), antialiased = False,
                  color = "lightskyblue", alpha = 0.6, shade = True)
axes.text2D(0.92, 0.83, "$\Sigma_{t + \delta t}$",
            transform = axes.transAxes, fontsize = 25.)



# ============
# Draw vectors
#    D---C
#    |   |
#    A---B
# ============
xA = 0.34
yA = 0.27
xB = 0.63
yB = 0.3
h  = 0.6

A = (xA, yA)
B = (xB, yB)
C = (xB, yB + h)
D = (xA, yA + h)


# \partial_t
axes.annotate("", xy = C, xytext = A,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = "red", edgecolor = "red",
                                width = 1., headwidth = 8.))
axes.text2D(0.42, 0.62, "$\partial_t$",
            transform = axes.transAxes, fontsize = 20., color = "red")

# \alpha n
axes.annotate("", xy = D, xytext = A,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                width = 1., headwidth = 8.))
axes.text2D(0.24, 0.6, "$\\alpha n$",
            transform = axes.transAxes, fontsize = 20., color = mydarkblue)
axes.annotate("4-velocity of the\nEulerian observer",
              xy = (0.3, 0.595), xytext = (-0.12, 0.46),
              fontsize = 12., fontweight = "bold", fontstyle = "normal",
              fontname = "Ubuntu", color = mydarkblue,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                arrowstyle = "->", connectionstyle = "arc3, rad = 0.4"),
              bbox = dict(boxstyle = "round, pad = 0.2", facecolor = mylightblue,
                          edgecolor = mydarkblue, linewidth = 2.))
axes.annotate("Lapse",
              xy = (0.25, 0.64), xytext = (0.01, 0.69),
              fontsize = 17., fontweight = "bold", fontstyle = "normal",
              fontname = "Ubuntu", color = mydarkblue,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                arrowstyle = "->", connectionstyle = "arc3, rad = -0.4"),
              bbox = dict(boxstyle = "round, pad = 0.2", facecolor = mylightblue,
                          edgecolor = mydarkblue, linewidth = 2.))

# \beta
axes.annotate("", xy = B, xytext = A,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                width = 1., headwidth = 8.))
axes.text2D(0.49, 0.22, "$\\beta$",
            transform = axes.transAxes, fontsize = 20., color = mydarkblue)
axes.annotate("Shift", xy = (0.54, 0.24), xytext = (0.78, 0.09),
              fontsize = 17., fontweight = "bold", fontstyle = "normal",
              fontname = "Ubuntu", color = mydarkblue,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                arrowstyle = "->", connectionstyle = "arc3, rad = 0.1"),
              bbox = dict(boxstyle = "round, pad = 0.2", facecolor = mylightblue,
                          edgecolor = mydarkblue, linewidth = 2.))

# Projection on `\alpha n' axis
axes.annotate("", xy = C, xytext = D,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                arrowstyle = "-", linestyle = "--"))

# Projection on `\Sigma_t' surface
axes.annotate("", xy = C, xytext = B,
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = mydarkblue, edgecolor = mydarkblue,
                                arrowstyle = "-", linestyle = "--"))

# `x^i = constant' curve
axes.annotate("", xy = A, xytext = (0.68, 0.82),
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = "red", edgecolor = "red",
                                arrowstyle = "-", connectionstyle = "arc3, rad = 0.08"))
axes.text2D(0.54, 0.57, "$x^i= constant$",
            transform = axes.transAxes, fontsize = 15., color = "red")

# 3-metric on each hypersurface
axes.text2D(0.22, 0.26, "$\gamma_{ij}$",
            transform = axes.transAxes, fontsize = 20., color = "darkgreen")
axes.annotate("3-metric", xy = (0.24, 0.24), xytext = (-0.03, 0.09),
              fontsize = 17., fontweight = "bold", fontstyle = "normal",
              fontname = "Ubuntu", color = "darkgreen",
              xycoords = "axes fraction", textcoords = "axes fraction",
              arrowprops = dict(facecolor = "darkgreen", edgecolor = "darkgreen",
                                arrowstyle = "->", connectionstyle = "arc3, rad = 0.3"),
              bbox = dict(boxstyle = "round, pad = 0.2", facecolor = "palegreen",
                          edgecolor = "darkgreen", linewidth = 2.))

"""
# Orthogonal projector on each hypersurface
axes.text2D(0.01, 0.13, "Orthogonal projector\n$P_{\mu\\nu} :\!\!= g_{\mu\\nu} - n_\mu n_\\nu$",
            transform = axes.transAxes, fontsize = 12., fontweight = "bold",
            fontstyle = "normal", fontname = "Ubuntu", color = "darkgreen")

# Extrinsic curvature
axes.text2D(0.01, 0.02, "Extrinsic curvature\n$K_{\mu\\nu} :\!\!= {P_\mu}^\\rho{P_\\nu}^\sigma\,\\nabla_\\rho n_\sigma = -\\frac{1}{2}\left(\mathcal{L}_n P\\right)_{\mu\\nu}$",
            transform = axes.transAxes, fontsize = 12., fontweight = "bold",
            fontstyle = "normal", fontname = "Ubuntu", color = "darkgreen")
"""



plt.tight_layout()
##plt.savefig("Foliation.pdf")
plt.savefig("Foliation.png")
plt.close()
