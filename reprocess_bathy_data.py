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

# %% [markdown]
# # Reprocess bathymetry data
#
# The purpose of this code is to take the existing data such as smith and sandwell, spamex and mbrud and make them suitable for merging. That means, reshaping to the right size domain (a little bigger than the model domain to make interpolation easy) and removing bad data points.
#
# First we load all the datasets and interpolate/resize them onto a fine spatial grid.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import ocean_tools.sandwell as ssb
import ocean_tools.utils as utils
import munch
import scipy.io as io

# DEFINE THE GRID USING GUNNARS GRID
data_dir = os.path.expanduser('~/data/SamoanPassage/Bathy/')
data_file = 'samoan_passage_bathymetry_200m_merged.mat'
B = munch.munchify(utils.loadmat(os.path.join(data_dir, data_file))['bathy2'])

lllon = B.lon.min()
lllat = B.lat.min()
urlon = B.lon.max()
urlat = B.lat.max()

# LOAD SMITH AND SANDWELL
dll = 0.25
SSX, SSY, SSD = ssb.read_grid([lllon-dll, urlon+dll, lllat-dll, urlat+dll])
SSX = SSX[:, 0]
SSY = SSY[0, :]
SSD = -1*SSD.T  # Transport necessary... want Y along i axis and X along j axis.

# Gunnars merged bathymetry
X = B.lon
Y = B.lat
Bmbrud = B.mbrud
Bspamex = B.spamex


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
# # First we look at the different batymetry datasets

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 2, figsize=(35, 20), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[1].contourf(SSX, SSY, SSD, clevs, cmap='Blues', extend='both')
axs[1].contour(SSX, SSY, SSD, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('Smith and Sandwell 19.1')
C = axs[0].contourf(X, Y, B.merged, clevs, cmap='Blues', extend='both')
axs[0].contour(X, Y, B.merged, clevs, linewidths=linewidths, colors='k')
axs[0].set_title("Gunnar's merged")

# %% [markdown]
# # The multibeam datasets

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(X, Y, Bspamex, clevs, cmap='Blues', extend='both')
axs[0].contour(X, Y, Bspamex, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('SPAMEX')
C = axs[1].contourf(X, Y, Bmbrud, clevs, cmap='Blues', extend='both')
axs[1].contour(X, Y, Bmbrud, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('MBRUD')

# %% [markdown]
# ## Because I don't know what version of Smith and Sandwell Gunnar used, lets reinterpolate that...

# %%
Xg, Yg = np.meshgrid(X, Y)
SShr = bilinear_interpolation(SSY, SSX, SSD, Yg.flatten(), Xg.flatten()).reshape(Xg.shape)

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(X, Y, SShr, clevs, cmap='Blues', extend='both')
axs[0].contour(X, Y, SShr, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('Smith and Sandwell Interpolated')
C = axs[1].contourf(SSX, SSY, SSD, clevs, cmap='Blues', extend='both')
axs[1].contour(SSX, SSY, SSD, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('Smith and Sandwell original')

# %% [markdown]
# ## Now let us fix some problems with the SPAMEX bathymetry

# %%
# Isolate weirdness
slllon = -168.5
slllat = -8.25
surlon = -168.3
surlat = -8.1
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

iis = np.searchsorted(X, (slllon, surlon))
ijs = np.searchsorted(Y, (slllat, surlat))
Bspamex_ = Bspamex.copy()
Bspamex_[ijs[0]:ijs[1], iis[0]:iis[1]] = np.nan

fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(X, Y, Bspamex, clevs, cmap='Blues', extend='both')
axs[0].contour(X, Y, Bspamex, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('spamex unedited')
C = axs[1].contourf(X, Y, Bspamex_, clevs, cmap='Blues', extend='both')
axs[1].contour(X, Y, Bspamex_, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('spamex edited')
axs[1].plot([slllon, slllon, surlon, surlon, slllon], [slllat, surlat, surlat, slllat, slllat], 'r')


# %%
# Spikes from gradient? This does seem very successful...

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GX, utils.mid(GY), np.log10(np.diff(bspamex, axis=0)))
C = axs[1].contourf(utils.mid(GX), GY, np.log10(np.diff(bspamex, axis=1)))
plt.colorbar(C)

# %%
# Very small holes are bad...

# bbs = 40  # How many points in a big square around the point to look at
# gd = np.isfinite(bspamex)
# bd = ~gd
# wspamex = weightgen(gd, bbs)**4

tofill = (wspamex > 0.8) & bd

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.pcolormesh(GX, GY, tofill)


# %% {"jupyter": {"outputs_hidden": true}}
idxs, jdxs = np.where(tofill)
bbs = 40
nr, nc = tofill.shape
for c in range(len(idxs)):
    i = idxs[c]
    j = jdxs[c]
    i1 = np.maximum(0, i-bbs)
    i2 = np.minimum(nr, i+bbs)
    j1 = np.maximum(0, j-bbs)
    j2 = np.minimum(nc, j+bbs)
    ig, jg = np.meshgrid(np.arange(i1, i2), np.arange(j1, j2))
    ig -= idxs[c]
    jg -= jdxs[c]
    dist2 = np.ma.masked_where(bd[i1:i2, j1:j2], ig**2 + jg**2)
    i_, j_ = np.unravel_index(np.argmin(dist2, axis=None), dist2.shape)
    print(Bspamex_[i_, j_])

# %% [markdown]
# ## Save the processed data

# %%
io.savemat('bathy_cleaned.mat', 
           {
               'lon': X,
               'lat': Y,
               'smith_sandwell': SShr,
               'mbrud': Bmbrud,
               'spamex': Bspamex_
           })
