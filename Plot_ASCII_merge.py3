import numpy as np
from matplotlib import pyplot as plt
##import scipy.interpolate as sc  # Needed for the commented code portion at the end


######################### USER-DEFINED PARAMETERS ##############################

qty = "qlm_spin[0]"
ext = "..asc"

t_col  = 8
ft_col = 12

time_units = "arbitrary"  # "arbitrary", "geometric" or "SI"

paths = np.array([
    "/scratch3/07825/lennoggi/BBH_handoff_McLachlan_pp08"
])

subdirs = np.array([
    "Scalars"
])

outputs = np.array([
    np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
])

normalize = False
absval    = False
log_scale = True

mytitle   = ""
my_ylabel = "$J\,\\left[M^2\\right]$"

plot_path = "/home1/07825/lennoggi"
plot_ext  = ".pdf"

colors = np.array([
    "dodgerblue"
])

labels = np.array([
    ""
])

################################################################################


# Sanity checks
N = len(paths)
assert(len(subdirs) == N)
assert(len(outputs) == N)
assert(len(colors)  == N)
assert(len(labels)  == N)

assert(normalize or not normalize)
assert(absval or not absval)
assert(log_scale or not log_scale)

# Set the time string
if time_units == "arbitrary":
    time_str = "$\left[M\\right]$"
elif time_units == "geometric":
    time_str = "$\left[M_{\!\!\odot}\!\\right]$"
elif time_units == "SI":
    time_str = "$\left[ms\\right]$"
else:
    raise RuntimeError("Invalid time units '" + time_units + "'. Please choose among 'arbitrary', 'geometric' and 'SI'.")


# Merge all the arrays to be plotted
t  = []
ft = []

for n in range(N):
    print("\n***** Simulation " + str(n + 1) + " *****")
    print("Path: " + paths[n])

    output_number = "output-" + str("{:0>4d}".format(outputs[n][0]))
    subdir        = subdirs[n] + "/" if subdirs[n] != "" else ""
    fullfname     = subdir + qty + ext
    fullpath      = paths[n] + "/" + output_number + "/" + fullfname
    data          = np.loadtxt(fullpath)

    t.append(data[:, t_col])
    tmax = t[-1][-1]

    if normalize:
        norm_fac = 1./(data[0, ft_col] - 1.)
        data.append(data[:, ft_col]*norm_fac)
    else:
        ft.append(data[:, ft_col])

    print(output_number + " merged")


    for i in outputs[n]:
        output_number = "output-" + str("{:0>4d}".format(i))
        fullpath      = paths[n] + "/" + output_number + "/" + fullfname
        data          = np.loadtxt(fullpath)
        data_arr      = data[:, t_col]
        L             = len(data_arr)

        for j in range(L):
            if data_arr[j] >= tmax:
                tmax_index = j
                break

        # If the previous for loop has been completed without being broken, it
        # means that all times in this output are smaller than tmax, so this
        # output should not be included  =>  Skip to next iteration now
        if j == L - 1:
            continue

        t[n] = np.concatenate((t[n],  data[tmax_index:, t_col]))
        tmax = t[-1][-1]

        if normalize:
            ft[n] = np.concatenate((ft[n], data[tmax_index:, ft_col]*norm_fac))
        else:
            ft[n] = np.concatenate((ft[n], data[tmax_index:, ft_col]))

        print(output_number + " merged")

    if (time_units == "SI"):
        t[n] *= 4.9257949707731345e-03



# Plot
plt.figure()
plt.title(mytitle, fontsize = 15., fontweight = "bold", fontstyle = "normal",
          fontname = "Ubuntu", color = "midnightblue")
plt.xlabel("$t\,$" + time_str, fontsize = 12.)
plt.ylabel(my_ylabel, fontsize = 12.)
##plt.xlim(1850., 4000.)
##plt.ylim(-0.000001, 0.000001)

if log_scale:
    plt.yscale("log")

for n in range(N):
    if absval:
        plt.plot(t[n], np.absolute(ft[n]),
                 linestyle = "-", linewidth = 1.,
                 marker = "", markersize = 2.,
                 color = colors[n], label = labels[n])
    else:
        plt.plot(t[n], ft[n],
                 linestyle = "-", linewidth = 1.,
                 marker = "", markersize = 2.,
                 color = colors[n], label = labels[n])

##plt.axvline(850., 0., 1., linestyle = "--", linewidth = 1.,
##    color = "red", label = "GW cross detector surface")

##plt.legend() ##(loc = "lower left", markerscale = 8.)
plt.tight_layout()
plt.savefig(plot_path + "/" + qty + plot_ext)
plt.close()



'''
# ***** Plot differences [LEAVE COMMENTED NORMALLY] *****
# The two arrays containing the times do NOT contain the same numbers (i.e., the
# same times) because different simulation restarts may overlap in time.
# Therefore, I need to interpolate the two GW signals to the same points before
# I can subtract them.

if len(ft[0]) >= len(ft[1]):
    time_arr = 1
else:
    time_arr = 0

x_array = np.linspace(t[time_arr][0], t[time_arr][-1], len(t[time_arr]))

func0 = sc.interp1d(t[0], ft[0])
func1 = sc.interp1d(t[1], ft[1])
diffs = func1(x_array) - func0(x_array)


# Plot differences
plt.figure()
##plt.title(mytitle, fontsize = 15., fontweight = "bold", fontstyle = "normal",
##    fontname = "Ubuntu", color = "midnightblue")
plt.xlabel("$t\,$" + time_str, fontsize = 12.)
plt.ylabel("$\Delta\psi_4\left(l = 2, m = 2\\right)$", fontsize = 12.)
plt.xlim(350., 2500.)
plt.ylim(-0.000005, 0.000005)
plt.plot(x_array, diffs,
         linestyle = "-", linewidth = 1., marker = "", markersize = 1.,
         color = "magenta")
plt.tight_layout()
plt.savefig("/home/lorenzo/Documents/Work/Talks and presentations/APS meeting April 2022/Plots/Diff_IGMdx0.125_Spritzdx0.125.pdf")
plt.close()


# Plot the Fourier transform of the difference between the signals
plt.figure()
##plt.title(mytitle, fontsize = 15., fontweight = "bold", fontstyle = "normal",
##    fontname = "Ubuntu", color = "midnightblue")
plt.xlabel("$\\nu$", fontsize = 12.)
plt.ylabel("$\mathcal{F}\left[\Delta\psi_4\left(l = 2, m = 2\\right)\\right]$", fontsize = 12.)
##plt.xlim(350., 2500.)
##plt.ylim(-0.000005, 0.000005)
plt.plot(np.fft.fftfreq(len(x_array)), np.fft.fft(diffs).real,
         linestyle = "-", linewidth = 1., marker = "", markersize = 1.,
         color = "dodgerblue")
plt.tight_layout()
plt.savefig("/home/lorenzo/Documents/Work/Talks and presentations/APS meeting April 2022/Plots/FT_Diff_IGMdx0.125_Spritzdx0.125.pdf")
plt.close()
'''
