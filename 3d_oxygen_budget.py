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