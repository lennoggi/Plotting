import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull


######################### USER-DEFINED PARAMETERS ##############################

# Iterations at which the AHs will be plotted
start = 11552
stop  = 13664
step  = 32

# Paths to AH data
AH1_dir = "/home/lorenzo/Scrivania/HDF5 files from simulations/Non-magnetised simulations for the TCAN/BNS_EOS_Sly4_1.4_1.4_Ab0.0_dx024_KastTh_full3D_LoreneInitData_LoreneETK_EvolveTfromBeginning_ET_2020_05_MARCONI/output-0002/Black hole/Apparent horizon 1/"
AH2_dir = "/home/lorenzo/Scrivania/HDF5 files from simulations/Non-magnetised simulations for the TCAN/BNS_EOS_Sly4_1.4_1.4_Ab0.0_dx024_KastTh_full3D_LoreneInitData_LoreneETK_EvolveTfromBeginning_ET_2020_05_MARCONI/output-0002/Black hole/Apparent horizon 2/"

# Plot destination and extension
plots_dir = "/home/lorenzo/Documenti/Tesi magistrale/Work/Apparent horizons/"
fig_ext   = ".pdf"

# Choose between Cactus or SI units of lenght.
# Cactus units: 1 solar mass distance = 1.476669691 km
space_units = "SI"  # "Cactus" or "SI"


# Plot setup
# ----------
# Plots will extend from -xylims to +xylims in both the x and y directions
xylims = 2. 
x1     = 0.065  # x coordinate of the bottom-left corner of plot 1
x2     = 0.565  # x coordinate of the bottom-left corner of plot 2
y      = 0.08   # y coordinate of the bottom-left corners of plots 1 and 2
xwidth = 0.42   # x-range of plots 1 and 2

title         = "SLy4_1.55_1.55"
titlefontsize = 35.

subtitle1        = "Apparent horizon 1"
subtitle2        = "Apparent horizon 2"
subtitlefontsize = 20.

labelfontsize = 15.

titlecolor   = "midnightblue"
myfontweight = "bold"
myfontstyle  = "normal"
myfontname   = "Ubuntu"

################################################################################



# Adjust plots based on the space units chosen (Cactus or SI)
if (space_units == "Cactus"):
    conv_fac_space = 1.
    unit_space_str = "[$\mathbf{M_{\\odot}}$]"
elif (space_units == "SI"):
    conv_fac_space = 1.476669691        # To get Km
    unit_space_str = "[$\mathbf{km}$]"
else:
    raise ValueError("Invalid space units. Either set them to \"Cactus\" or \"SI\"")



################################### PLOT #######################################
for it in range(start, stop + 1, step):
    try:
        data1 = np.loadtxt(AH1_dir + "h.t" + str(it) + ".ah1.gp")
        data2 = np.loadtxt(AH2_dir + "h.t" + str(it) + ".ah2.gp")

        # Find the convex hull of the projections of the points in the apparent
        # horizon in the xy plane
        hull1  = ConvexHull(data1[:, [3, 4]])
        xhull1 = data1[:, 3][hull1.vertices]
        yhull1 = data1[:, 4][hull1.vertices]

        hull2  = ConvexHull(data2[:, [3, 4]])
        xhull2 = data2[:, 3][hull2.vertices]
        yhull2 = data2[:, 4][hull2.vertices]


        # 12.8*300 = 3840, 7.2*300 = 2160  =>  3840×2160 16:9 4K frame
        fig         = plt.figure(figsize = [12.8, 7.2], dpi = 300)
        ywidth      = xwidth*16./9.
        xylims_conv = xylims*conv_fac_space
        

        # Plot 1
        ax = fig.add_axes([x1, y, xwidth, ywidth])
        ax.set_title(subtitle1, y = 1.01, color = titlecolor,
                     fontsize  = subtitlefontsize, fontweight = myfontweight,
                     fontstyle = myfontstyle,      fontname   = myfontname)
        ax.set_xlabel("x" + unit_space_str, fontsize = labelfontsize)
        ax.set_ylabel("y" + unit_space_str, fontsize = labelfontsize,
                      labelpad = -5.)
        ax.set_xlim(-xylims_conv, xylims_conv)
        ax.set_ylim(-xylims_conv, xylims_conv)
        ax.fill(xhull1*conv_fac_space, yhull1*conv_fac_space,
                linewidth = 0., facecolor = "black")
        plt.grid(linestyle = "--")


        # Plot 2
        ax = fig.add_axes([x2, y, xwidth, ywidth])
        ax.set_title(subtitle2, y = 1.01, color = titlecolor,
                     fontsize  = subtitlefontsize, fontweight = myfontweight,
                     fontstyle = myfontstyle,      fontname   = myfontname)
        ax.set_xlabel("x" + unit_space_str, fontsize = labelfontsize)
        ax.set_ylabel("y" + unit_space_str, fontsize = labelfontsize,
                      labelpad = -5.)
        ax.set_xlim(-xylims_conv, xylims_conv)
        ax.set_ylim(-xylims_conv, xylims_conv)
        ax.fill(xhull2*conv_fac_space, yhull2*conv_fac_space,
                linewidth = 0., facecolor = "black")
        plt.grid(linestyle = "--")


        # Set figure title and time and iteration info
        fig.text(x1, 0.92, title, color = titlecolor, fontsize = titlefontsize,
                 weight = myfontweight, fontstyle = myfontstyle,
                 fontname = myfontname)

        it_str = "It = " + str(it)
        fig.text(x2, 0.92, it_str, color = "red", fontsize = 25.,
                 fontweight = myfontweight, fontstyle = myfontstyle,
                 fontname   = myfontname)


        # Save the plot and close the figure
        plt.savefig(plots_dir + "AH" + str(it) + fig_ext)
        plt.close()


    # Catch a possible exception if the file is not found
    except OSError as file_not_found:
        print(file_not_found)
        print("\n")
