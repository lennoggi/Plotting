from matplotlib import pyplot as plt, patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


fig = plt.figure(dpi = 600)
ax1 = fig.add_subplot(121, projection = "3d")
ax2 = fig.add_subplot(122, projection = "3d")

ax1.view_init(elev = 15.) ##, azim = 270.)
ax2.view_init(elev = 15.) ##, azim = 270.)


ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])


##ax1.set_xlabel("$\log\left(r\\right)$", fontsize = 9., labelpad = -12.)
##ax1.set_ylabel("$\\theta$", labelpad = -10., rotation = 180.)
##ax1.set_zlabel("$\phi$",    labelpad = -10.)
ax1.text2D(0.17, 0.11, "$\log\left(r\\right)$", transform = ax1.transAxes, rotation = -10.5)
ax1.text2D(0.84, 0.18, "$\\theta$",             transform = ax1.transAxes)
ax1.text2D(0.99, 0.53, "$\phi$",                transform = ax1.transAxes)

ax2.set_xlabel("$x$", labelpad = -12.)
ax2.text2D(0.86, 0.18, "$y$", transform = ax2.transAxes)
ax2.text2D(1.,   0.53, "$z$", transform = ax2.transAxes)


ax1.text2D(-0.06, 0.19, "$\log\left(15\\right)$",    fontsize = 6., transform = ax1.transAxes, color = "blue")
ax1.text2D( 0.42, 0.09, "$\log\left(20000\\right)$", fontsize = 6., transform = ax1.transAxes, color = "blue")
ax1.text2D( 0.68, 0.11, "$0$",                       fontsize = 6., transform = ax1.transAxes, color = "red")
ax1.text2D( 0.95, 0.27, "$\pi$",                     fontsize = 6., transform = ax1.transAxes, color = "red")
ax1.text2D( 0.96, 0.31, "$0$",                       fontsize = 6., transform = ax1.transAxes, color = "forestgreen")
ax1.text2D( 0.97, 0.75, "$2\pi$",                    fontsize = 6., transform = ax1.transAxes, color = "forestgreen")


ax1.text2D(0.21, 0.9, "Original domain",
           fontsize = 12., fontweight = "bold", fontstyle = "normal", fontname = "Ubuntu",
           color = "Midnightblue", transform = ax1.transAxes)
ax2.text2D(0.17, 0.9, "Destination domain",
           fontsize = 12., fontweight = "bold", fontstyle = "normal", fontname = "Ubuntu",
           color = "Midnightblue", transform = ax2.transAxes)


arrow = patches.FancyArrowPatch((0.33, 0.5), (0.8, 0.5), transform = fig.transFigure,
                                facecolor = "black", connectionstyle = "arc3, rad=-0.25", arrowstyle = "<|-", mutation_scale = 20.)
fig.patches.append(arrow)

ax1.text2D(0.5, 0.52, "$\mathbf{x}$", transform = ax1.transAxes, fontsize = 15., fontweight = "bold")
ax1.text2D(0.55, 0.49, ".", transform = ax1.transAxes, fontsize = 15., fontweight = "bold")

ax2.text2D(0.74, 0.52, "$\mathbf{x'}$", transform = ax2.transAxes, fontsize = 15., fontweight = "bold")
ax2.text2D(0.72, 0.49, ".", transform = ax2.transAxes, fontsize = 15., fontweight = "bold")


plt.tight_layout()
plt.savefig("Coordinate_mapping_sph2Cart.pdf")
##plt.savefig("Coordinate_mapping_sph2Cart.png")
plt.close()
