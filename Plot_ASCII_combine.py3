import numpy as np
from matplotlib import pyplot as plt


# ************************* USER-DEFINED PARAMETERS ****************************

filename1 = "/scratch3/07825/lennoggi/BBH_analysis/CactusVolumeIntegrals/CactusVolumeIntegrals_run_NonSpinning_large_TaperedCooling_r15/Volume_integrals.asc"
filename2 = "/scratch3/07825/lennoggi/BBH_analysis/CactusAnalysis/CactusAnalysis_run_NonSpinning_large_TaperedCooling_r15/Surface_integrals.asc"

# Supported operations:
# "sum":       a + b
# "sub":       a - b
# "mult":      a*b
# "ratio":     a/b
# "ratio2":    a/(b*b)
# "ratiosqrt": a/sqrt(b)
operation = "ratio"

tcols     = (1, 1)
fcols     = (7, 3)
negatives = (False, True)  # Use the negative data? Useful e.g. with the accretion rate

ylabel  = "$\\frac{L_{EM}}{\dot{M}}$"
figname = "EM_efficiency_NonSpinning_large_TaperedCooling_r15.pdf"

# ******************************************************************************


# Data loading and sanity checks
data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)
assert data1.shape[0] == data2.shape[0]

t  = data1[:, tcols[0]]
t2 = data2[:, tcols[1]]
assert np.array_equal(t2, t)

f1 = -data1[:, fcols[0]] if negatives[0] else data1[:, fcols[0]]
f2 = -data2[:, fcols[1]] if negatives[1] else data2[:, fcols[1]]


# Combine the data
if   operation == "sum":       ftot = f1 + f2
elif operation == "sub":       ftot = f1 - f2
elif operation == "mult":      ftot = f1*f2
elif operation == "ratio":     ftot = f1/f2
elif operation == "ratio2":    ftot = f1/(f2*f2)
elif operation == "ratiosqrt": ftot = f1/np.sqrt(f2)
else: raise RuntimeError("Invalid operation '" + operation + "'")

# Plot
fig = plt.figure(figsize = [10., 4.])
plt.xlabel("$t\,\left[M\\right]$", fontsize = 12.)
plt.ylabel(ylabel, fontsize = 12.)
plt.grid(linestyle = "--", linewidth = 0.5, alpha = 0.5)
plt.ylim(1.e-02, 6.)
plt.yscale("log")
plt.axvline(12079., linestyle = "--", linewidth = 1., color = "black")
bbox_props  = dict(boxstyle = "round", linewidth = 1.,
                   ##facecolor = "powderblue", edgecolor = "midnightblue")
                   facecolor = "powderblue", edgecolor = "midnightblue", alpha = 0.5)
plt.text(0.785, 0.88, "Merger", fontsize = 10., fontweight = "bold",
         fontstyle = "normal", fontfamily = "Ubuntu", color = "midnightblue",
         multialignment = "center", bbox = bbox_props, transform = fig.gca().transAxes)
plt.plot(t, ftot, linestyle = "-", linewidth = 1., marker = "", color = "mediumslateblue")
plt.tight_layout()
plt.savefig(figname)
plt.close()

print("Plot saved as '" + figname + "'")
