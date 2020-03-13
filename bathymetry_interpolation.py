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
from numba import jit
import warnings

warnings.filterwarnings("ignore")

# LLC4320 bathymetry
depth = np.fromfile('grid/Depth_144x185', '>f').reshape((185, 144))
XC = np.fromfile('grid/XC_144x185', '>f').reshape((185, 144))
YC = np.fromfile('grid/YC_144x185', '>f').reshape((185, 144))

# Smith and Sandwell bathymetry from whichever file I have
dll = 0.25
SSX, SSY, SSD = ssb.read_grid([XC.min()-dll, XC.max()+dll, YC.min()-dll, YC.max()+dll])
SSD *= -1

# Gunnars merged bathymetry
data_dir = os.path.expanduser('~/data/SamoanPassage/Bathy/')
data_file = 'samoan_passage_bathymetry_200m_merged.mat'
B = munch.munchify(utils.loadmat(os.path.join(data_dir, data_file))['bathy2'])
iis = np.searchsorted(B.lon, (XC.min(), XC.max()))
ijs = np.searchsorted(B.lat, (YC.min(), YC.max()))
GX = B.lon[iis[0]:iis[1]]
GY = B.lat[ijs[0]:ijs[1]]
GD = B.merged[ijs[0]:ijs[1], iis[0]:iis[1]]
mbrud = B.mbrud[ijs[0]:ijs[1], iis[0]:iis[1]]
bspamex = B.spamex[ijs[0]:ijs[1], iis[0]:iis[1]]


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
# # First we look at the different batymetry datasets

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 3, figsize=(35, 20), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(XC, YC, depth, clevs, cmap='Blues', extend='both')
axs[0].contour(XC, YC, depth, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('LLC4320')
C = axs[1].contourf(SSX, SSY, SSD, clevs, cmap='Blues', extend='both')
axs[1].contour(SSX, SSY, SSD, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('Smith and Sandwell 19.1')
C = axs[2].contourf(GX, GY, GD, clevs, cmap='Blues', extend='both')
axs[2].contour(GX, GY, GD, clevs, linewidths=linewidths, colors='k')
axs[2].set_title("Gunnar's merged")

# %% [markdown]
# # A bigger version of Gunnars merged dataset

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.set_aspect('equal')
C = ax.contourf(GX, GY, GD, clevs, cmap='Blues', extend='both')
ax.contour(GX, GY, GD, clevs, linewidths=linewidths, colors='k')

# %% [markdown]
# # The multibeam datasets

# %%
dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, axs = plt.subplots(1, 2, figsize=(20, 20), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GX, GY, bspamex, clevs, cmap='Blues', extend='both')
axs[0].contour(GX, GY, bspamex, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('SPAMEX')
# Isolate weirdness
lllon = -168.5
lllat = -8.27
urlon = -168.3
urlat = -8.1
axs[0].plot([lllon, lllon, urlon, urlon, lllon], [lllat, urlat, urlat, lllat, lllat], 'r')
C = axs[1].contourf(GX, GY, mbrud, clevs, cmap='Blues', extend='both')
axs[1].contour(GX, GY, mbrud, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('MBRUD')

# %% [markdown]
# ## Clean up SPAMEX

# %%
# Isolate weirdness
lllon = -168.5
lllat = -8.25
urlon = -168.3
urlat = -8.1

iis = np.searchsorted(GX, (lllon, urlon))
ijs = np.searchsorted(GY, (lllat, urlat))
bspamex[ijs[0]:ijs[1], iis[0]:iis[1]] = np.nan

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')

C = ax.contourf(GX, GY, bspamex, clevs, cmap='Blues', extend='both')
ax.contour(GX, GY, bspamex, clevs, linewidths=linewidths, colors='k')
ax.set_title('SPAMEX')
# Isolate weirdness
lllon = -168.5
lllat = -8.27
urlon = -168.3
urlat = -8.1
ax.plot([lllon, lllon, urlon, urlon, lllon], [lllat, urlat, urlat, lllat, lllat], 'r')


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


# %%
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
    print(bspamex[i_, j_])

# %% [markdown]
# ## Lets test out bilinear interpolation

# %%
loni = np.linspace(-170., -169.5, 1000)
lati = np.linspace(-9.3, -9, 1000)
bt = bilinear_interpolation(GX, GY, B.spamex[ijs[0]:ijs[1], iis[0]:iis[1]], loni, lati)

dcon = 50
linewidths = 0.05
clevs = np.arange(3500, 5500+dcon, dcon)

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
ax.set_aspect('equal')
C = ax.contourf(GX, GY, bspamex, clevs, cmap='Blues', extend='both')
ax.contour(GX, GY, bspamex, clevs, linewidths=linewidths, colors='k')
ax.plot(loni, lati, 'r')

fig, ax = plt.subplots(1, 1)
ax.plot(lati, bt)
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
# ## Now we want to create the best possible bathymetry...
#
# To start we use, as a basis, the original model bathymetry. We must interpolate this onto a much finer grid... the one Gunnar used is a good start.

# %%
GXg, GYg = np.meshgrid(GX, GY)
B1 = bilinear_interpolation(YC[:, 0], XC[0, :], depth, GYg.flatten(), GXg.flatten()).reshape(GYg.shape)
# B1SS = bilinear_interpolation(SSY[:, 0], SSX[0, :], SSD, GYg.flatten(), GXg.flatten()).reshape(GYg.shape)

# %%
# Compare the two... they should be identical looking.
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GXg, GYg, B1, clevs, cmap='Blues', extend='both')
axs[0].contour(GXg, GYg, B1, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('Interpolated')
C = axs[1].contourf(XC, YC, depth, clevs, cmap='Blues', extend='both')
axs[1].contour(XC, YC, depth, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('Original')

# %% [markdown]
# ## Develop a merging algorithm...

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

# %% [markdown]
# ## Try merging the mbrud with MITgcm...

# %%
# w = (np.tanh(2*weight*np.pi - np.pi)/2 + 1/2)**2
w = weight**4
B2 = w*mbrud + (1 - w)*B1
B2[np.isnan(B2)] = B1[np.isnan(B2)]

# %%
# Compare the two... they should look different
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GXg, GYg, B1, clevs, cmap='Blues', extend='both')
axs[0].contour(GXg, GYg, B1, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('MITgcm depth interpolated to Gunnar grid')
C = axs[1].contourf(GXg, GYg, B2, clevs, cmap='Blues', extend='both')
axs[1].contour(GXg, GYg, B2, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('MITgcm depth merged with mbrud')

# %% [markdown]
# ## Try merging with SPAMEX bathymetry

# %%
bbs = 40  # How many points in a big square around the point to look at
gd = np.isfinite(bspamex)
weight1 = weightgen(gd, bbs)

# %%
step = 4
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(GXg[::step, ::step], GYg[::step, ::step], weight1[::step, ::step])
plt.colorbar(C)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
C = ax.pcolormesh(GXg[::step, ::step], GYg[::step, ::step], weight1[::step, ::step]**4)
plt.colorbar(C)  # (np.tanh(2*weight[::step, ::step]*np.pi - np.pi)/2 + 1/2)**2

# %%
w = weight1**4
B3 = w*bspamex + (1 - w)*B2
B3[np.isnan(B3)] = B2[np.isnan(B3)]

# %%
# Compare the two... they should look different
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GXg, GYg, B2, clevs, cmap='Blues', extend='both')
axs[0].contour(GXg, GYg, B2, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('MITgcm depth merged with mbrud')
C = axs[1].contourf(GXg, GYg, B3, clevs, cmap='Blues', extend='both')
axs[1].contour(GXg, GYg, B2, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('MITgcm depth merged with mbrud and spamex')

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey=True)

for ax in axs:
    ax.set_aspect('equal')

C = axs[0].contourf(GXg, GYg, B2, clevs, cmap='Blues', extend='both')
axs[0].contour(GXg, GYg, B2, clevs, linewidths=linewidths, colors='k')
axs[0].set_title('MITgcm depth merged with mbrud')
C = axs[1].contourf(GXg, GYg, B3, clevs, cmap='Blues', extend='both')
axs[1].contour(GXg, GYg, B3, clevs, linewidths=linewidths, colors='k')
axs[1].set_title('MITgcm depth merged with mbrud and spamex')
