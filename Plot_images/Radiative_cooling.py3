from matplotlib import pyplot as plt

# ***** User-defined parameters *****
sep      = 0.6
xAH      = 0.5*sep
r_cavity = 1.5*sep
r_max    = 5.4 ##1.8*sep

merger_sep = 0.7

rcool_min_inspiral = 0.35
rcool_max_inspiral = 1.

rcool_min_merger = 0.45
rcool_max_merger = 1.
# ***********************************


fig = plt.figure(dpi = 600)
ax  = fig.gca()

# Fill the CBD region
xcoords = [-r_max,  r_max, r_max, -r_max, -r_max]
ycoords = [-r_max, -r_max, r_max,  r_max, -r_max]
plt.fill(xcoords, ycoords, "dodgerblue")

# Fill circular regions
if sep <= merger_sep:
    print("Merger")
    ax.add_artist(plt.Circle((  0., 0.), rcool_max_merger, linestyle = "", fill = True, color = "orange"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_min_merger, linestyle = "", fill = True, color = "black"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_min_merger, linestyle = "", fill = True, color = "black"))
else:
    print("Inspiral")
    ax.add_artist(plt.Circle((  0., 0.), r_cavity,           linestyle = "", fill = True, color = "paleturquoise"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "orange"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "orange"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))

plt.xlim(-r_max, r_max)
plt.ylim(-r_max, r_max)
ax.set_aspect("equal")
plt.tight_layout()
##plt.savefig("Radiative_cooling.pdf")
plt.savefig("Radiative_cooling.jpg")
plt.close()
