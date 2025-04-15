
import os
data_folder = "Data"
cuprite_nevada_folder = "Cuprite Nevada"

filename = "ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.TIF"

raster_path = os.path.join(data_folder, cuprite_nevada_folder, filename)

from Raster import Raster

raster = Raster(path=raster_path)


import geopandas as gpd
from vector_utils import clip_raster
import utils

polygon_path = os.path.join(data_folder, cuprite_nevada_folder, "ROI.geojson")

polygon = gpd.read_file(polygon_path)
raster = clip_raster(raster, polygon)

raster = utils.preprocess(raster)

from Spectrum import Spectrum

ref_spectrum = Spectrum(mineral_name="kaolinite")
ref_spectrum.preprocess(desired_wavelengths=raster.wavelength)

import matplotlib.pyplot as plt

plt.plot(ref_spectrum.wavelength, ref_spectrum.reflectance)
plt.title("Kaolinite")
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflectance")


import numpy as np

sam_score = utils.spectralMatch(raster, ref_spectrum, method="sam")
threshold = 0.07
masked_sam_score = np.ma.masked_greater(sam_score, threshold)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Plot the results
# RGB from the hyperspectral image
raster_for_rgb = Raster(path=raster_path)
raster_for_rgb = clip_raster(raster_for_rgb, polygon)

rgb_indices = utils.get_rgb_indices(raster_for_rgb)

red = raster_for_rgb.datacube[rgb_indices[0], :, :]
green = raster_for_rgb.datacube[rgb_indices[1], :, :]
blue = raster_for_rgb.datacube[rgb_indices[2], :, :]

raster_for_rgb.datacube = np.stack([red, green, blue])

# Normalize the RGB datacube
raster_data = raster_for_rgb.datacube
raster_data = raster_data.astype(float)  # Convert to float for normalization
raster_data /= raster_data.max()  # Normalize to 0-1

# Rearrange the axes to (height, width, channels)
rgb_image = np.transpose(raster_data, (1, 2, 0))

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(rgb_image)

sam_im = ax.imshow(masked_sam_score, cmap='turbo_r')

# Create a colorbar with a title and labels
norm = Normalize(vmin=0, vmax=1)
mapper = ScalarMappable(norm=norm, cmap='turbo')
# colorbar = plt.colorbar(mapper, fraction=0.02, pad=0.04)
colorbar = plt.colorbar(mapper, ax=ax, fraction=0.02, pad=0.04)

colorbar.set_label('Kaolinite')
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['Low', 'High'])

ax.set_title('Kaolinite Probability Map')

fig.savefig('map.svg', format='svg')

plt.show()


row, col = 230, 260

target_spec = raster.datacube[:, row, col]

fig, ax = plt.subplots()
ax.plot(raster.wavelength, target_spec, label="Target")
ax.plot(raster.wavelength, ref_spectrum.reflectance, label="Reference Spectrum - Kaolinite")

ax.legend(frameon=False)
ax.set_xlabel("Wavelength (um)")
ax.set_ylabel("Reflectance")
ax.set_title(f"Spectral Comparison at Row {row}, Column {col}")

fig.savefig('spectrum.svg', format='svg')

plt.show()


