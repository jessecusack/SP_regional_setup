# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: spamex
#     language: python
#     name: spamex
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import ocean_tools.utils as utils
import scipy.interpolate as itpl

depth = np.fromfile('grid/Depth_144x185', '>f').reshape((185, 144))
XC = np.fromfile('grid/XC_144x185', '>f').reshape((185, 144))
YC = np.fromfile('grid/YC_144x185', '>f').reshape((185, 144))
hFacC = np.fromfile('grid/hFacC_144x185x90', '>f').reshape((90, 185, 144))
T = np.fromfile('Theta_144x185x90.0000808704', '>f').reshape((90, 185, 144))

# %% [markdown]
# ## Min and max depths of high resolution

# %%
# Max depth
dmax = np.max(depth)
print("Max depth {}".format(dmax))

# 88 cells
delR = np.array([1.00,    1.14,    1.30,    1.49,   1.70,
          1.93,    2.20,    2.50,    2.84,   3.21,
          3.63,    4.10,    4.61,    5.18,   5.79,
          6.47,    7.20,    7.98,    8.83,   9.73,
         10.69,   11.70,   12.76,   13.87,  15.03,
         16.22,   17.45,   18.70,   19.97,  21.27,
         22.56,   23.87,   25.17,   26.46,  27.74,
         29.00,   30.24,   31.45,   32.65,  33.82,
         34.97,   36.09,   37.20,   38.29,  39.37,
         40.45,   41.53,   42.62,   43.73,  44.87,
         46.05,   47.28,   48.56,   49.93,  51.38,
         52.93,   54.61,   56.42,   58.38,  60.53,
         62.87,   65.43,   68.24,   71.33,  74.73,
         78.47,   82.61,   87.17,   92.21,  97.79,
        103.96,  110.79,  118.35,  126.73, 136.01,
        146.30,  157.71,  170.35,  184.37, 199.89,
        217.09,  236.13,  257.21,  280.50, 306.24,
        334.64,  365.93,  400.38])

z = np.hstack((0, np.cumsum(delR)))

# np.unravel_index(np.argmin(a, axis=None), a.shape)

T_ = T.copy()
T_[hFacC < 1] = 1e10
T_[T > 1] = 1e10
Tm1 = T_ - 1
idxs = np.argmin(np.abs(Tm1), axis=0)
idxs[(T_ == 1e10).all(axis=0)] = 10000
imin = np.min(idxs)
dmin = utils.mid(z)[imin]
print("Min depth {}".format(dmin))

fig, ax = plt.subplots(1, 1)
ax.plot(delR, utils.mid(z), '.')
ax.invert_yaxis()
ax.axhline(dmax, color='r')
ax.axhline(dmin, color='r')


# %% [markdown]
# # Make a new z

# %%
# 90 cells
delR =  np.array([ 1.00,    1.14,    1.30,    1.49,   1.70,
          1.93,    2.20,    2.50,    2.84,   3.21,
          3.63,    4.10,    4.61,    5.18,   5.79,
          6.47,    7.20,    7.98,    8.83,   9.73,
         10.69,   11.70,   12.76,   13.87,  15.03,
         16.22,   17.45,   18.70,   19.97,  21.27,
         22.56,   23.87,   25.17,   26.46,  27.74,
         29.00,   30.24,   31.45,   32.65,  33.82,
         34.97,   36.09,   37.20,   38.29,  39.37,
         40.45,   41.53,   42.62,   43.73,  44.87,
         46.05,   47.28,   48.56,   49.93,  51.38,
         52.93,   54.61,   56.42,   58.38,  60.53,
         62.87,   65.43,   68.24,   71.33,  74.73,
         78.47,   82.61,   87.17,   92.21,  97.79,
        103.96,  110.79,  118.35,  126.73, 136.01,
        146.30,  157.71,  170.35,  184.37, 199.89,
        217.09,  236.13,  257.21,  280.50, 306.24,
        334.64,  365.93,  400.38,  438.23, 479.74])

z = np.hstack((0, np.cumsum(delR)))

fig, ax = plt.subplots(1, 1)
ax.plot(z)

# %%
dhrmin = 3600.
dhrmax = 6000.
dd = 15.
dhr = np.arange(dhrmin, dhrmax+dd, dd)

fig, ax = plt.subplots(1, 1)
ax.plot(delR, utils.mid(z), '.')
ax.plot(delR[57:59], utils.mid(z)[57:59], '.')
ax.invert_yaxis()
ax.axhline(dmax, color='r')
ax.axhline(dmin, color='r')

ax.plot(np.diff(dhr), utils.mid(dhr), '.')
ax.plot(np.diff(dhr)[:2], utils.mid(dhr)[:2], '.')

# %%
f = itpl.interp1d(np.hstack((utils.mid(z)[57:59], utils.mid(dhr)[:2])), np.hstack((delR[57:59], np.diff(dhr)[:2])), kind='cubic')


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(delR, utils.mid(z), '.')
ax.plot(delR[57:59], utils.mid(z)[57:59], '.')
ax.invert_yaxis()
ax.axhline(dmax, color='r')
ax.axhline(dmin, color='r')

ax.plot(f(np.linspace(1302, 3600)), np.linspace(1302, 3600))

ax.plot(np.diff(dhr), utils.mid(dhr), '.')
ax.plot(np.diff(dhr)[:2], utils.mid(dhr)[:2], '.')

ax.set_xlim(0, 100)
