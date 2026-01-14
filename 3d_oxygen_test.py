# For testing only, developing...

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

# Define grid
lon = np.linspace(-180, 180, 60)
lat = np.linspace(-90, 90, 30)
depth = np.array([0, 100, 300, 700, 1500, 3000])

# Mock physical fields
temp = np.random.uniform(0, 30, size=(len(depth), len(lat), len(lon)))
salt = np.random.uniform(34, 36, size=(len(depth), len(lat), len(lon)))
oxygen = np.random.uniform(100, 300, size=(len(depth), len(lat), len(lon)))

# === MOCK O2 Saturation ===
# Simplified physical model: high in cold, low in warm
O2_sat = 350 - 5 * temp + 0.1 * (salt - 35)

# Apparent Oxygen Utilization
aou = O2_sat - oxygen

# Interpolation across depth
depth_dense = np.linspace(depth.min(), depth.max(), 80)
aou_interp_func = RegularGridInterpolator(
    (depth, lat, lon), aou, bounds_error=False, fill_value=np.nan
)

# Generate full 3D grid
lon3d, lat3d, depth3d = np.meshgrid(lon, lat, depth_dense, indexing="ij")
points_interp = np.array([depth3d.flatten(), lat3d.flatten(), lon3d.flatten()]).T
aou_interp = aou_interp_func(points_interp).reshape(len(lon), len(lat), len(depth_dense))

# PyVista UniformGrid setup
spacing = (
    lon[1] - lon[0],
    lat[1] - lat[0],
    depth_dense[1] - depth_dense[0],
)
origin = (lon[0], lat[0], -depth_dense[-1])  # Invert Z for surface at top

grid = pv.UniformGrid()
grid.dimensions = np.array(aou_interp.shape) + 1  # dimensions are points
grid.origin = origin
grid.spacing = spacing
grid.cell_data["AOU"] = aou_interp.flatten(order="F")

# Plot
plotter = pv.Plotter()
plotter.add_volume(
    grid,
    scalars="AOU",
    cmap="viridis",
    opacity="linear",
    shade=True,
    scalar_bar_args={"title": "AOU (µmol/kg, mock)"},
)
plotter.add_axes()
plotter.show_grid()
plotter.show()
