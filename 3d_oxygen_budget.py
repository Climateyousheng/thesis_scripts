import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import xarray as xr
import os
import cftime
import gsw
import pyvista as pv

# I/O setup
dir_in = '/Users/nd20983/Documents/data/'
data_list = ['xpvha','xpvhb','xpvhc','xpvhr','xpvhw',]
dscb_list = ['GETECH','Scotese','Robertsons','GET_RC','GET_LC']
file_tail = 'o.pgclann.nc'
dir_out = os.path.abspath(os.path.join(
    os.getcwd(), '..','..','Simulations','data','OBM','figures'))
os.makedirs(dir_out, exist_ok=True)
# dir_outpath = os.path.join(dir_out, f"AOU_map_{i}.png")

# read in files
ds_list = []
for data in data_list:
    filename = os.path.join(dir_in, data + file_tail)
    # non-standard time units
    ds = xr.open_dataset(filename, decode_times=False)
    # manually convert time to cftime
    ds['t'].attrs['units'] = 'days since 2850-12-01 00:00:00'
    ds = xr.decode_cf(ds)
    ds_list.append(ds)

# # loop over ds_list
# for i, ds in enumerate(ds_list):
#     # get essential variables
#     temp = ds['temp_ym_dpth']
#     salt = ds['salinity_ym_dpth']
#     oxygen = ds['O2_ym_dpth']
#     waterage = ds['agewater_ym_dpth']
#     # get coordinates
#     lat = ds['latitude']
#     lon = ds['longitude']
#     depth = ds['depth_2']

#     # compute oxygen solubility (in umol/kg)
#     O2_sat = gsw.O2sol_SP_pt(salt.values, temp.values)

#     # compute AOU
#     aou = xr.DataArray(O2_sat - oxygen.values, coords=oxygen.coords, dims=oxygen.dims)
#     aou.name = 'AOU'
#     aou.attrs["units"] = "µmol/kg"
#     aou.attrs["long_name"] = "Apparent Oxygen Utilization"

#     # add to dataset
#     ds['AOU'] = aou

#     # plots for vertical layers
#     rows, cols =5, 4
#     fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14,12), constrained_layout=True,
#                             subplot_kw={'projection': ccrs.Mollweide()})
#     for d, ax in zip(range(rows * cols), axs.ravel()):
#         im = ax.pcolormesh(lon, lat, aou.isel(t=0, depth_2=d), shading='auto', cmap='viridis',
#                            vmin=0, vmax=350, transform=ccrs.PlateCarree())
#         ax.set_title(f"AOU at {int(depth[d].values)} m", fontsize=10)
#     # plt.colorbar(im, label='AOU (µmol/kg)')
#     fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical', label='AOU (µmol/kg)',
#                  shrink=0.8, aspect=30)
#     fig.suptitle(dscb_list[i], fontsize=14)

#     # export figure
#     # dir_outpath = os.path.join(dir_out, f"AOU_depth_layers_{dscb_list[i]}.png")
#     # fig.savefig(dir_outpath, dpi=300,)

#     # plots for zonal means
#     # Ocean basin masks
#     basins = {
#         "Atlantic": ((lon >= -100) & (lon <= 20)),
#         "Pacific": ((lon >= 120) | (lon <= -70)),
#         "Indian": ((lon >= 20) & (lon <= 120)),
#     }

#     fig1, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#     # Zonal mean plots
#     for j, (name, mask) in enumerate(basins.items()):
#         ax = axs[j]
#         aou_basin = aou.where(mask, drop=True)
#         zonal_mean = aou_basin.mean(dim="longitude", skipna=True)

#         im = zonal_mean.plot(ax=ax, y="depth_2", yincrease=False, cmap="viridis", 
#                              vmin=0, vmax=350, cbar_kwargs={"label": "AOU (µmol/kg)"})
#         ax.set_title(f"{name} Ocean")
#         ax.set_xlabel("Latitude")
#         # for first column only set up y label
#         if j == 0:
#             ax.set_ylabel("Depth (m)")
#         else:
#             ax.set_ylabel("")

#     # fig.colorbar(im, ax=axs.ravel().tolist(), label="AOU (µmol/kg)", orientation='vertical')

#     fig1.suptitle(f"Zonal Mean AOU by Ocean Basin in {dscb_list[i]}", fontsize=14)

#     # export figure
#     # dir_outpath1 = os.path.join(dir_out, f"AOU_zonal_mean_ocean_basins_{dscb_list[i]}.png")
#     # fig1.savefig(dir_outpath1, dpi=300)

# get essential variables
ds = ds_list[0]

temp = ds['temp_ym_dpth']
salt = ds['salinity_ym_dpth']
oxygen = ds['O2_ym_dpth']
waterage = ds['agewater_ym_dpth']
# get coordinates
lat = ds['latitude']
lon = ds['longitude']
depth = ds['depth_2']

# compute oxygen solubility (in umol/kg)
O2_sat = gsw.O2sol_SP_pt(salt.values, temp.values)

# compute AOU
aou = xr.DataArray(O2_sat - oxygen.values, coords=oxygen.coords, dims=oxygen.dims)
aou.name = 'AOU'
aou.attrs["units"] = "µmol/kg"
aou.attrs["long_name"] = "Apparent Oxygen Utilization"

# add to dataset
ds['AOU'] = aou
aou = ds['AOU'].isel(t=0)

# Get coordinate arrays
lon = ds['longitude'].values
lat = ds['latitude'].values
depth = ds['depth_2'].values

# Downsample to reduce size
aou = aou.isel(longitude=slice(None, None, 4), latitude=slice(None, None, 4))
lon = lon[::4]
lat = lat[::4]

# Meshgrid in lon-lat plane
lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')

# Create a uniform grid for each depth slab
plotter = pv.Plotter()
for k in range(len(depth) - 1):
    z_top = -depth[k]
    z_bot = -depth[k+1]
    z_mid = (z_top + z_bot) / 2

    # AOU at layer k (constant over slab thickness)
    aou_layer = aou.isel(depth_2=k).values

    # Expand to 3D by stacking top and bottom
    nx, ny = aou_layer.shape
    x = lon2d.flatten()
    y = lat2d.flatten()
    z = np.full_like(x, z_mid)

    points = np.column_stack((x, y, z*0.04))
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [nx, ny, 1]
    grid["AOU"] = aou_layer.flatten()

    surf = grid.cast_to_unstructured_grid().extract_surface()
    plotter.add_mesh(surf, scalars="AOU", cmap="viridis", show_edges=False, opacity=1)

# Final display
plotter.add_axes()
plotter.show_grid()
plotter.show()