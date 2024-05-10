from matplotlib import pyplot as plt
from matplotlib import figure

# ***** User-defined parameters *****
sep        = 0.85 ##0.6 ##0.85 ##5.
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
    ax.add_artist(plt.Circle((0., 0.), rcool_max_merger, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle((0., 0.), rcool_min_merger, linestyle = "", fill = True, color = "black"))
    """
    plt.hlines(0.,                     0., rcool_min_merger, linestyle = "-",  linewidth = 1.5, color = "white")
    plt.hlines(-0.02*rcool_min_merger, 0., rcool_max_merger, linestyle = "-",  linewidth = 1.5, color = "mediumblue")
    plt.text(0.23*rcool_min_merger, 0.05*rcool_min_merger, "0.45", fontsize = 8.,  fontweight = "bold", color = "white")
    plt.text( 0.7*rcool_min_merger, 0.05*rcool_min_merger, "M",    fontsize = 8.,  fontweight = "bold", color = "white")
    plt.text(1.05*rcool_min_merger, -0.2*rcool_min_merger, "1",    fontsize = 8.,  fontweight = "bold", color = "mediumblue")
    plt.text(1.18*rcool_min_merger, -0.2*rcool_min_merger, "M",    fontsize = 8.,  fontweight = "bold", color = "mediumblue")
    """
else:
    print("Inspiral")
    ax.add_artist(plt.Circle((  0., 0.), r_cavity,           linestyle = "", fill = True, color = "paleturquoise"))
    ax.add_artist(plt.Circle(( xAH, 0.), r_minidisk,         linestyle = "", fill = True, color = "limegreen"))
    ax.add_artist(plt.Circle((-xAH, 0.), r_minidisk,         linestyle = "", fill = True, color = "limegreen"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_max_inspiral, linestyle = "", fill = True, color = "darkorange"))
    ax.add_artist(plt.Circle(( xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))
    ax.add_artist(plt.Circle((-xAH, 0.), rcool_min_inspiral, linestyle = "", fill = True, color = "black"))
    """
    plt.arrow(-xAH, 0.,  2.*xAH, 0., linestyle = "-", linewidth = 1.5, facecolor = "red", edgecolor = "red",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    plt.arrow( xAH, 0., -2.*xAH, 0., linestyle = "-", linewidth = 1.5, facecolor = "red", edgecolor = "red",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    plt.arrow( xAH, 0., r_minidisk, 0., linestyle = "-", linewidth = 1.5, facecolor = "mediumblue", edgecolor = "mediumblue",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    plt.arrow( xAH + r_minidisk, 0., -r_minidisk, 0., linestyle = "-", linewidth = 1.5, facecolor = "mediumblue", edgecolor = "mediumblue",
              head_width = 0.1, head_length = 0.1, length_includes_head = True)
    """
    """
    plt.hlines(0., -xAH, xAH,              linestyle = "-", linewidth = 1.5, color = "red")
    plt.hlines(0.,  xAH, xAH + r_minidisk, linestyle = "-", linewidth = 1.5, color = "mediumblue")
    plt.text(-0.04*sep, 0.03*sep, "a",    fontsize = 10., fontweight = "bold", color = "red")
    plt.text( 0.71*sep, 0.03*sep, "0.45", fontsize = 8.,  fontweight = "bold", color = "mediumblue")
    plt.text( 0.94*sep, 0.03*sep, "a",    fontsize = 10., fontweight = "bold", color = "mediumblue")
    """
    """
    plt.hlines(-0.015*sep, xAH, xAH + rcool_max_inspiral, linestyle = "-", linewidth = 1.5, color = "blue")
    plt.hlines(0., xAH, xAH + rcool_min_inspiral, linestyle = "-",  linewidth = 1.5, color = "white")
    plt.text(0.55*sep,  0.03*sep, "0.35", fontsize = 8.,  fontweight = "bold", color = "white")
    plt.text(0.79*sep,  0.03*sep, "M",    fontsize = 8.,  fontweight = "bold", color = "white")
    plt.text(0.95*sep, -0.12*sep, "1",    fontsize = 8.,  fontweight = "bold", color = "blue")
    plt.text(1.01*sep, -0.12*sep, "M",    fontsize = 8.,  fontweight = "bold", color = "blue")
    """


plt.xlim(-r_max, r_max)
plt.ylim(-r_max, r_max)
ax.set_aspect("equal", adjustable = "box")
##plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
plt.tight_layout()
##plt.savefig("Radiative_cooling.pdf")
plt.savefig("Radiative_cooling.jpg")
plt.close()
