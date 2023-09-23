import numpy as np
from matplotlib import pyplot as plt


######################### USER-DEFINED PARAMETERS ##############################

filename1 = "/home1/07825/lennoggi/qlm_spin[0].asc"
filename2 = "/home1/07825/lennoggi/qlm_mass[0].asc"

# Supported operations:
# 1: a + b
# 2: a - b
# 3: a*b
# 4: a/b
# 5: a/(b*b)
operation = 5

t_col  = 0
ft_col = 1

plot_title    = "++0.8 run, apparent horizon 0"
my_ylabel     = "$\chi$"
plot_fullpath = "/home1/07825/lennoggi/qlm_dimensionless_spin[0]_pp08.pdf"

################################################################################


data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)

assert(len(data1) == len(data2))

t  = data1[:, 0]
f1 = data1[:, 1]
f2 = data2[:, 1]

N = len(t)

for n in range(N):
    assert(data2[n, 0] == t[n])

assert(len(f1) == N)
assert(len(f2) == N)


if operation == 1:
    ftot = f1 + f2
elif operation == 2:
    ftot = f1 - f2
elif operation == 3:
    ftot = f1*f2
elif operation == 4:
    ftot = f1/f2
elif operation == 5:
    ftot = f1/(f2*f2)
else:
    raise RuntimeError("Invalid operation '" + operation + "'")


plt.figure()
plt.title(plot_title, fontsize = 15., fontweight = "bold",
          fontstyle = "normal", fontname = "Ubuntu", color = "midnightblue")
plt.xlabel("$t\,\left[M\\right]$")
##plt.xlabel("$t\,\left[ms\\right]$")
plt.ylabel(my_ylabel)
plt.plot(t, ftot,
##plt.plot(t*4.9257949707731345e-03, ft,
         linestyle = "-", marker = ".", markersize = 3., color = "dodgerblue")
plt.tight_layout()
plt.savefig(plot_fullpath)
plt.close()
