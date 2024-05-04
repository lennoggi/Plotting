import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def Gaussian_exp(x, C, mu, xi, sigma2, q, tail):
    if   (tail == "l"):  return C*np.exp(xi*(x - mu + xi*sigma2/2.))*(1 - scipy.special.erf((x - mu + xi*sigma2)/(np.sqrt(2)*sigma2))) + q
    elif (tail == "r"):  return C*np.exp(-xi*(x - mu - xi*sigma2/2.))*(1 + scipy.special.erf((x - mu - xi*sigma2)/(np.sqrt(2)*sigma2))) + q
    else:                raise RuntimeError("Please set 'tail' to either \"l\" or \"r\"")



######################### USER-DEFINED PARAMETERS ##############################
C      = 1.
mu     = 7.
xi     = 0.5
sigma2 = 0.5
q      = 0.05

tail = "l"  # "l" for left tail, "r" for right tail

m_char1  = 0.9
xc_char1 = 6.6

m_char2  = 4.
xc_char2 = 7.6

x         = np.linspace(-0.3, 10., 100000)
ymax_char = 0.8
x_char1   = np.linspace(xc_char1, xc_char1 + ymax_char/(1.*m_char1), 100000)
x_char2   = np.linspace(xc_char2, xc_char2 + ymax_char/(1.*m_char2), 100000)

shift = 0.04
################################################################################



fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)


# ======
# Plot 1
# ======
ax1.set_xlabel("$x$", fontsize = 15., x = 0.98)
ax1.set_ylabel("$u_0\left(x\\right)$", fontsize = 15., y = 0.93, labelpad = 25., rotation = 0.)

ax1.tick_params(axis = "x", which = "both", bottom = False, labelbottom = False)
ax1.tick_params(axis = "y", which = "both", left   = False, labelleft = False)

ax1.spines["left"].set_position("zero")
ax1.spines["bottom"].set_position("zero")
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)

ax1.plot(1, 0, ">k", transform = ax1.get_yaxis_transform(), clip_on = False)
ax1.plot(0, 1, "^k", transform = ax1.get_xaxis_transform(), clip_on = False)

ax1.plot(x, Gaussian_exp(x, C, mu, xi, sigma2, q, tail), linestyle = "-", linewidth = 1., marker = "", color = "red")

ax1.vlines(xc_char2, 0., Gaussian_exp(xc_char2, C, mu, xi, sigma2, q, tail),
           linestyle = "--", linewidth = 0.7, color = "black")
ax1.hlines(Gaussian_exp(xc_char2, C, mu, xi, sigma2, q, tail), 0., xc_char2,
           linestyle = "--", linewidth = 0.7, color = "black")

ax1.hlines(Gaussian_exp(xc_char1, C, mu, xi, sigma2, q, tail), 0., xc_char1,
           linestyle = "--", linewidth = 0.7, color = "black")
ax1.vlines(xc_char1, 0., Gaussian_exp(xc_char1, C, mu, xi, sigma2, q, tail),
           linestyle = "--", linewidth = 0.7, color = "black")

ax1.text(xc_char1 - 0.1, -0.10, "$x_1$", fontsize = 10.)
ax1.text(xc_char2 - 0.1, -0.10, "$x_2$", fontsize = 10.)
ax1.text(-0.9, Gaussian_exp(xc_char1, C, mu, xi, sigma2, q, tail) - 0.02,
         "$u_0\left(x_1\\right)$", fontsize = 10.)
ax1.text(-0.9, Gaussian_exp(xc_char2, C, mu, xi, sigma2, q, tail) - 0.02,
         "$u_0\left(x_2\\right)$", fontsize = 10.)
ax1.annotate("", xy = (10., 0.7), xytext = (7.5, 0.7),
             arrowprops = dict(facecolor = "black", arrowstyle="-|>"))
ax1.text(7.9, 0.8, "Direction of\npropagation", fontsize = 9.)



# ======
# Plot 2
# ======
ax2.set_xlabel("$x$", fontsize = 15., x = 0.98)
ax2.set_ylabel("$t$", fontsize = 15., y = 0.93, labelpad = 15., rotation = 0.)
ax2.set_ylim(0., 1.)

ax2.tick_params(axis = "x", which = "both", bottom = False, labelbottom = False)
ax2.tick_params(axis = "y", which = "both", left   = False, labelleft   = False)

ax2.spines["left"].set_position("zero")
ax2.spines["bottom"].set_position(("axes", shift))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax2.plot(1., shift, ">k", transform = ax2.get_yaxis_transform(), clip_on = False)
ax2.plot(0., 1.,    "^k", transform = ax2.get_xaxis_transform(), clip_on = False)

ax2.plot(x_char1, m_char1*(x_char1 - xc_char1) + shift,
         linestyle = '-', linewidth = 1., marker = "", color = "royalblue")
ax2.plot(x_char2, m_char2*(x_char2 - xc_char2) + shift,
         linestyle = '-', linewidth = 1., marker = "", color = "royalblue")

ax2.axvline(xc_char1, shift, 1., linestyle = "--", linewidth = 0.7, color = "black")
ax2.axvline(xc_char2, shift, 1., linestyle = "--", linewidth = 0.7, color = "black")
ax2.text(xc_char1 - 0.1, -0.07 + shift, "$x_1$", fontsize = 10.)
ax2.text(xc_char2 - 0.1, -0.07 + shift, "$x_2$", fontsize = 10.)



plt.tight_layout()
plt.savefig("Shock_wave.pdf")
plt.close()
