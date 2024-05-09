from matplotlib import pyplot as plt
from matplotlib import figure

# ***** User-defined parameters *****
sep        = 5. ##0.6 ##0.85 ##6.
xAH        = 0.5*sep
r_minidisk = 0.45*sep
r_cavity   = 1.5*sep
r_max      = 1.8*sep ##1.8*0.85 ##1.8*sep  ##10.8

merger_sep = 0.7

rcool_min_inspiral = 0.35
rcool_max_inspiral = 1.

rcool_min_merger = 0.45
rcool_max_merger = 1.
# ***********************************


fig = plt.figure(figsize = figure.figaspect(0.9), dpi = 600)
ax  = fig.gca()

# Fill the CBD region
xcoords = [-r_max,  r_max, r_max, -r_max, -r_max]
ycoords = [-r_max, -r_max, r_max,  r_max, -r_max]
plt.fill(xcoords, ycoords, "dodgerblue")

# Fill circular regions
if sep <= merger_sep:
    print("Merger")
    ax.add_artist(plt.Circle((  0., 0.), rcool_max_merger, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_min_merger, linestyle = "", fill = True, color = "black"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_min_merger, linestyle = "", fill = True, color = "black"))
else:
    print("Inspiral")
    ax.add_artist(plt.Circle((  0., 0.), r_cavity,           linestyle = "", fill = True, color = "paleturquoise"))
    ax.add_artist(plt.Circle(( xAH, 0.), r_minidisk,         linestyle = "", fill = True, color = "limegreen"))
    ax.add_artist(plt.Circle((-xAH, 0.), r_minidisk,         linestyle = "", fill = True, color = "limegreen"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))
    plt.arrow(-xAH, 0., 2.*xAH, 0., linestyle = "-", linewidth = 1.5, facecolor = "red", edgecolor = "red",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    plt.arrow(xAH, 0., -2.*xAH, 0., linestyle = "-", linewidth = 1.5, facecolor = "red", edgecolor = "red",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    plt.text(-0.05*sep, 0.03*sep, "$a$", fontsize = 13., fontweight = "bold", color = "red")


plt.xlim(-r_max, r_max)
plt.ylim(-r_max, r_max)
ax.set_aspect("equal", adjustable = "box")
##plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
plt.tight_layout()
##plt.savefig("Radiative_cooling.pdf")
plt.savefig("Radiative_cooling.jpg")
plt.close()
