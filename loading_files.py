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
U = np.fromfile('U_144x185x90.0000808704', '>f').reshape((90, 185, 144))
V = np.fromfile('V_144x185x90.0000808704', '>f').reshape((90, 185, 144))
T = np.fromfile('Theta_144x185x90.0000808704', '>f').reshape((90, 185, 144))

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')
C = ax.contourf(XC, YC, depth, np.arange(3500, 5750, 250), cmap='Blues', extend='both')
plt.colorbar(C)

# %%
step = 4

fT = (T < 1).astype(float)
fh = (hFacC == 1).astype(float)

use = (T < 1) & (hFacC == 1)
UM = np.ma.masked_where(~use, U).mean(axis=0)
VM = np.ma.masked_where(~use, V).mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')
C = ax.contourf(XC, YC, depth, np.arange(3500, 5750, 250), cmap='Blues', extend='both')
Q = ax.quiver(XC[::step, ::step], YC[::step, ::step], UM[::step, ::step], VM[::step, ::step])

plt.colorbar(C)
plt.quiverkey(Q, 0.8, 0.1, 0.2, 'U m/s', coordinates='axes')

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(U[:, 0, 0])
ax.plot(V[:, 0, 0])
print(XC[0, 0], YC[0, 0])
