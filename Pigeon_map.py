import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import array_bounds, xy
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import warnings
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Input raster
# ----------------------------
tif = "rocpig_abundance_seasonal_year_round_mean_2023.tif"
src = rasterio.open(tif)

# Read raster data (masked values -> NaN)
data = src.read(1, masked=True)
data_np = np.where(data.mask, np.nan, data.data)

if src.crs is None:
    raise ValueError("Raster has no CRS. Cannot reliably overlay vector data. Check the .prj file.")

# ----------------------------
# 1) Reproject raster to EPSG:4326 if needed
# ----------------------------
dst_crs = "EPSG:4326"
if src.crs.to_string() != dst_crs:
    print("Reprojecting raster to EPSG:4326 ... (may be slow)")
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    dst = np.full((dst_height, dst_width), np.nan, dtype=data_np.dtype)
    reproject(
        source=data_np,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )
    data_np = dst
    plot_transform = dst_transform
    plot_w, plot_h = dst_width, dst_height
else:
    plot_transform = src.transform
    plot_w, plot_h = src.width, src.height

# ----------------------------
# 2) Compute lon/lat bounds
# ----------------------------
minx, miny, maxx, maxy = array_bounds(plot_h, plot_w, plot_transform)
print("Computed lon/lat bounds:", (minx, miny, maxx, maxy))

# ----------------------------
# 3) Detect 0-360 lon and fix
# ----------------------------
if minx >= 0 and maxx > 180:
    print("Detected 0..360 lon range -> converting to -180..180 by rolling columns.")
    cols = np.arange(plot_w)
    xs, _ = xy(plot_transform, np.zeros(plot_w, dtype=int), cols, offset="center")
    xs = np.array(xs)
    idx = np.where(xs > 180)[0]
    if idx.size > 0:
        shift = idx[0]  # roll this column to left
        data_np = np.roll(data_np, -shift, axis=1)
        new_minx = xs[shift] - 360.0
        new_maxx = new_minx + (maxx - minx)
        minx, maxx = new_minx, new_maxx
        print(f"Rolled by {shift} columns -> new lon bounds: ({minx}, {maxx})")
    else:
        print("No columns found with xs>180, skipping 0-360 correction.")

extent = [minx, maxx, miny, maxy]
print("Final extent (left, right, bottom, top):", extent)

# ----------------------------
# 4) Determine origin
# ----------------------------
origin = "upper" if plot_transform.e < 0 else "lower"
print("Using origin =", origin, "; transform.e =", plot_transform.e)

# ----------------------------
# 5) Colormap: set NaN to transparent
# ----------------------------
cmap = get_cmap("viridis")  # can switch to 'cividis' or custom for journal style
cmap.set_bad("none")

# ----------------------------
# 6) Compute robust vmin/vmax (2nd and 98th percentile)
# ----------------------------
vmin, vmax = np.nanpercentile(data_np, [2, 98])

# ----------------------------
# 7) Load world boundaries (ensure EPSG:4326)
# ----------------------------
world = gpd.read_file("worldcountries.shp")
if world.crs is None:
    world = world.set_crs("EPSG:4326")
else:
    world = world.to_crs("EPSG:4326")

# =============================
# Plotting
# =============================
# Parameters
zero_color = "#d9d9d9"
cmap_list = ["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"]
cmap_main = LinearSegmentedColormap.from_list("custom_bluegreen", cmap_list)
tol_zero = 1e-12

# Masks
mask_nodata = np.isnan(data_np)
mask_zero = (~mask_nodata) & (np.isclose(data_np, 0, atol=tol_zero))
mask_nonzero = (~mask_nodata) & (~mask_zero)
data_main = np.where(mask_nonzero, data_np, np.nan)

# Main value range
if np.count_nonzero(mask_nonzero) == 0:
    vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = np.nanpercentile(data_main, [2, 98])

# Create figure
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.Robinson())

# Crop to lat > -60° to exclude Antarctica
ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

# Background and coastlines
ax.add_feature(cfeature.OCEAN, color="#f5f7fa", zorder=0)
ax.add_feature(cfeature.LAND, color="#fbfbfb", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="gray", zorder=6)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 8, "color": "gray"}
gl.ylabel_style = {"size": 8, "color": "gray"}
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

# Main map (non-zero values)
cmap_main.set_bad("none")
main_img = ax.imshow(
    data_main,
    origin="upper",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap=cmap_main,
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
    zorder=2,
)

# Zero value layer
zero_layer = np.where(mask_zero, 1.0, np.nan)
cmap_zero = ListedColormap([zero_color])
ax.imshow(
    zero_layer,
    origin="upper",
    extent=extent,
    transform=ccrs.PlateCarree(),
    cmap=cmap_zero,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    zorder=3,
)

# Country boundaries
sf = ShapelyFeature(world.geometry, ccrs.PlateCarree(),
                    edgecolor="gray", facecolor="none", linewidth=0.4)
ax.add_feature(sf, zorder=4)

# Continuous colorbar
cbar = plt.colorbar(main_img, ax=ax, shrink=0.55, pad=0.03)
cbar.set_label("Relative abundance (Rock Pigeon, 2023)", fontsize=10)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8)

# Title
plt.title("Global distribution — Rock Pigeon (Columba livia, 2023)",
          fontsize=14, weight="bold", pad=12)

# Save high-resolution figure (publication quality)
plt.tight_layout()
plt.savefig(
    "rock_pigeon_distribution_highres_trimmed.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)
plt.show()

