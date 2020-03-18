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
import os
import numpy as np
import matplotlib.pyplot as plt
import ocean_tools.sandwell as ssb
import ocean_tools.utils as utils
import munch

# LLC4320 bathymetry
depth = np.fromfile('grid/Depth_144x185', '>f').reshape((185, 144))
XC = np.fromfile('grid/XC_144x185', '>f').reshape((185, 144))
YC = np.fromfile('grid/YC_144x185', '>f').reshape((185, 144))

bathy = np.fromfile('grid/BATHY_144x185_Box13', '>f').reshape((185, 144))

MB = munch.munchify(utils.loadmat('merged_bathy.mat'))

def bilinear_interpolation(xa, ya, fg, x, y):
    i2 = np.searchsorted(xa, x)
    i1 = i2 - 1
    j2 = np.searchsorted(ya, y)
    j1 = j2 - 1
    dx = xa[i2] - xa[i1]
    dy = ya[j2] - ya[j1]
    f11, f21, f12, f22 = fg[i1, j1], fg[i2, j1], fg[i1, j2], fg[i2, j2]
    x1, y1, x2, y2 = xa[i1], ya[j1], xa[i2], ya[j2]
    fi = (f11*(x2 - x)*(y2 - y) + f21*(x - x1)*(y2 - y) +
          f12*(x2 - x)*(y - y1) + f22*(x - x1)*(y - y1))/(dx*dy)
    return fi



# %% [markdown]
# ## Lets test out bilinear interpolation

# %%
loni = np.linspace(-170., -168.5, 1000)
lati = np.linspace(-9.3, -8, 1000)
BTSS = bilinear_interpolation(MB.lat, MB.lon, MB.SS_only, lati, loni)
BTSSMS = bilinear_interpolation(MB.lat, MB.lon, MB.SS_mbrud_spamex, lati, loni)

dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.set_aspect('equal')
C = ax.contourf(MB.lon, MB.lat, MB.SS_mbrud_spamex, clevs, cmap='Blues', extend='both')
ax.contour(MB.lon, MB.lat, MB.SS_mbrud_spamex, clevs, linewidths=linewidths, colors='k')
ax.plot(loni, lati, 'r')

fig, ax = plt.subplots(1, 1)
ax.plot(lati, BTSS, label='Smith and Sandwell')
ax.plot(lati, BTSSMS, label='All merged')
ax.invert_yaxis()

# %% [markdown]
# ## Now let us draw a transition zone around the model bathymetry...

# %%
nt = 10
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')
C = ax.contourf(XC, YC, depth, clevs, cmap='Blues', extend='both')
ax.contour(XC, YC, depth, clevs, linewidths=linewidths, colors='k')
ax.plot([XC[nt, nt], XC[nt, -nt], XC[-nt, -nt], XC[-nt, nt], XC[nt, nt]], 
        [YC[nt, nt], YC[nt, -nt], YC[-nt, -nt], YC[-nt, nt], YC[nt, nt]],
       color='r')

lllon = XC.min() + 0.25
lllat = YC.min() + 0.25
urlon = XC.max() - 0.25
urlat = YC.max() - 0.25
ax.plot([lllon, lllon, urlon, urlon, lllon], [lllat, urlat, urlat, lllat, lllat], 'y')

# %% [markdown]
# ### Define weight function based on transition zone

# %%
nt = 15
idxs = np.indices(depth.shape)
wt = np.minimum(idxs[0, ...]/nt, 1.)
wl = np.minimum(idxs[1, ...]/nt, 1.)
w = wl*wt*wl[:, ::-1]*wt[::-1, :]

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
C = ax.pcolormesh(XC, YC, w)
plt.colorbar(C)


# %%
fig, axs = plt.subplots(1, 2)
axs[0].plot(YC[:, 30], w[:, 30])
axs[1].plot(XC[60, :], w[60, :])

# %% [markdown]
# ## Now we have to interpolate high res onto model grid...

# %%
BI = bilinear_interpolation(MB.lat, MB.lon, MB.SS_only, YC.flatten(), XC.flatten()).reshape(XC.shape)

# %%
new_depth = w*BI + (1-w)*depth

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(XC, YC, depth, clevs, cmap='Blues', extend='both')
axs[0].contour(XC, YC, depth, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('original')
C = axs[1].contourf(XC, YC, new_depth, clevs, cmap='Blues', extend='both')
axs[1].contour(XC, YC, new_depth, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('new')

fig, ax = plt.subplots(1, 1)
C = ax.pcolormesh(XC, YC, depth - new_depth, cmap='coolwarm', vmin=-500, vmax=500)
cb = plt.colorbar(C)
cb.set_label('Difference (m)')

# %% [markdown]
# ## Save new bathymetry
#
# Also double check that it works...
#
# We need the minus sign because the input bathy should be negative!!!! We also need to make sure the big endian float 4 is correct.

# %%
((-1*new_depth).astype('>f')).tofile('out/bathy.bin')

# %%
test_ = -np.fromfile('out/bathy.bin', '>f').reshape((185, 144))

fig, ax = plt.subplots(1, 1)
C = ax.pcolormesh(XC, YC, new_depth - test_, cmap='coolwarm', vmin=-500, vmax=500)
cb = plt.colorbar(C)
cb.set_label('Difference (m)')
