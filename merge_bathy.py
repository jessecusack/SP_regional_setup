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
# # We want to create the best possible bathymetry...

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import ocean_tools.utils as utils
import munch
from numba import jit
import scipy.io as io

B = munch.munchify(utils.loadmat('bathy_cleaned.mat'))

@jit
def weightgen(gd, bbs):
    w = np.zeros_like(gd, dtype=float)
    bbmax = (2*bbs)**2
    nr, nc = gd.shape
    for i in range(nr):
        for j in range(nc):
            i1 = np.maximum(0, i-bbs)
            i2 = np.minimum(nr, i+bbs)
            j1 = np.maximum(0, j-bbs)
            j2 = np.minimum(nc, j+bbs)
            w[i, j] = gd[i1:i2, j1:j2].sum()/bbmax
    return w


# %% [markdown]
# ## Try merging the mbrud with Smith and Sandwell

# %%
#  w = (np.tanh(2*weight*np.pi - np.pi)/2 + 1/2)**2
bbs = 40
gd = np.isfinite(B.mbrud)
B1 = B.smith_sandwell
weight1 = weightgen(gd, bbs)
w = weight1**4
B2 = w*B.mbrud + (1 - w)*B1
B2[np.isnan(B2)] = B1[np.isnan(B2)]

# %%
# Compare the two... they should look different
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(B.lon, B.lat, B1, clevs, cmap='Blues', extend='both')
axs[0].contour(B.lon, B.lat, B1, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('Smith and Sandwell interpolated to high resolution')
C = axs[1].contourf(B.lon, B.lat, B2, clevs, cmap='Blues', extend='both')
axs[1].contour(B.lon, B.lat, B2, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('Smith and Sandwell merged with mbrud')

step = 4
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(B.lon[::step], B.lat[::step], w[::step, ::step])
plt.colorbar(C)

# %% [markdown]
# ## Try merging all bathymetry as well as SS with spamex only

# %%
bbs = 40
gd = np.isfinite(B.spamex)
weight2 = weightgen(gd, bbs)
w = weight2**4
B3 = w*B.spamex + (1 - w)*B2
B3[np.isnan(B3)] = B2[np.isnan(B3)]
B4 = w*B.spamex + (1 - w)*B1
B4[np.isnan(B4)] = B1[np.isnan(B4)]

# %%
# Compare the two... they should look different
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(B.lon, B.lat, B1, clevs, cmap='Blues', extend='both')
axs[0].contour(B.lon, B.lat, B1, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('SS interpolated to high resolution')
C = axs[1].contourf(B.lon, B.lat, B3, clevs, cmap='Blues', extend='both')
axs[1].contour(B.lon, B.lat, B3, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('SS merged with mbrud and spamex')
C = axs[2].contourf(B.lon, B.lat, B4, clevs, cmap='Blues', extend='both')
axs[2].contour(B.lon, B.lat, B4, clevs, linewidths=linewidths, colors='k')
axs[2].set_title('SS merged with spamex only')

step = 4
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(B.lon[::step], B.lat[::step], w[::step, ::step])
plt.colorbar(C)

# %% [markdown]
# ## Save the merged bathymetry!

# %%
io.savemat('merged_bathy.mat', 
           {
               'lon': B.lon,
               'lat': B.lat,
               'SS_only': B.smith_sandwell,
               'SS_mbrud': B2,
               'SS_spamex': B4,
               'SS_mbrud_spamex': B3,
           })

# %% [markdown]
# ## Merging algorithm development

# %%
# Let us see if we can find edges...
# Split topography into good (with data) and bad (nan)
gd = np.isfinite(mbrud)
bd = ~gd

shift = -10
edgex1 = ~(gd ^ np.roll(bd, shift=shift, axis=0))
edgey1 = ~(gd ^ np.roll(bd, shift=shift, axis=1))

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(edgex1 | edgey1)

# %%
# Counting NaN method
bbs = 40  # How many points in a big square around the point to look at
gd = np.isfinite(mbrud)
weight = weightgen(gd, bbs)

# %%
step = 4
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(GXg[::step, ::step], GYg[::step, ::step], weight[::step, ::step])
plt.colorbar(C)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(GXg[::step, ::step], GYg[::step, ::step], weight[::step, ::step]**4)
plt.colorbar(C)  # (np.tanh(2*weight[::step, ::step]*np.pi - np.pi)/2 + 1/2)**2

# %% [markdown]
# ## Different weight scalings
# We need this to get rid of the sharp edges... I think weight of 0.5 should be squished to near zero.

# %%
w = np.linspace(0, 1, 100)
fig, ax = plt.subplots(1, 1)
ax.plot(w, w)
ax.plot(w, w**2)
ax.plot(w, w**3)
ax.plot(w, w**4)
ax.plot(w, (np.tanh(2*w*np.pi - np.pi)/2 + 1/2))
ax.plot(w, (np.tanh(2*w*np.pi - np.pi)/2 + 1/2)**2)
