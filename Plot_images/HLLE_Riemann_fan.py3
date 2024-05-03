import numpy as np
from matplotlib import pyplot as plt


xl = np.linspace(-0.65, 0., 2)
xr = np.linspace(0., 1., 2)

sl = -1.7
sr = 1.1


fig = plt.figure()
plt.xlabel("$x$", fontsize = 20., x = 0.97)
plt.ylabel("$t$", fontsize = 20., y = 0.96, labelpad = 15., rotation = 0.)
##plt.xlim(-1.5, 1.7)

plt.tick_params(axis = "x", which = "both", bottom = False, labelbottom = False)
plt.tick_params(axis = "y", which = "both", left   = False, labelleft   = False)

fig.axes[0].spines["left"].set_position("zero")
fig.axes[0].spines["bottom"].set_position("zero")
fig.axes[0].spines["right"].set_visible(False)
fig.axes[0].spines["top"].set_visible(False)

plt.plot(1, 0, ">k", transform = fig.axes[0].get_yaxis_transform(), clip_on = False)
plt.plot(0, 1, "^k", transform = fig.axes[0].get_xaxis_transform(), clip_on = False)
plt.plot(xl, sl*xl, linestyle = '-', linewidth = 2., marker = "", color = "royalblue")
plt.plot(xr, sr*xr, linestyle = '-', linewidth = 2., marker = "", color = "royalblue")

plt.hlines(1., -1.5, 1.8, linestyles = "--", linewidths = 1., color = "black")
plt.vlines(-1.3, 0., 1.,  linestyles = "--", linewidths = 1., color = "black")
plt.vlines(-0.59, 0., 1., linestyles = "--", linewidths = 1., color = "black")
plt.vlines(0.91, 0., 1.,  linestyles = "--", linewidths = 1., color = "black")
plt.vlines(1.4, 0., 1.,   linestyles = "--", linewidths = 1., color = "black")


plt.annotate("$u_L$",       xy = (-1, 0.5),       fontsize = 25., color = "royalblue")
plt.annotate("$u_{HLLE}$",  xy = (0.1, 0.85),     fontsize = 25., color = "royalblue")
plt.annotate("$u_R$",       xy = (1., 0.3),       fontsize = 25., color = "royalblue")
plt.annotate("0",           xy = (0.02, -0.04),   fontsize = 10.)
plt.annotate("$x_L$",       xy = (-1.35, -0.05),  fontsize = 15.)
plt.annotate("$x_R$",       xy = (1.35, -0.05),   fontsize = 15.)
plt.annotate("$t\sigma_L$", xy = (-0.65, -0.055), fontsize = 15.)
plt.annotate("$t\sigma_R$", xy = (0.84, -0.055),  fontsize = 15.)
plt.annotate("$T$",         xy = (0.05, 1.02),    fontsize = 15.)

plt.tight_layout()
plt.savefig("HLLE_Riemann_fan.pdf")
plt.close()
