"""
HSI Preprocessing and Visualization
- Reads HSI data
- Extracts wavelengths from metadata
- Selects SWIR bands for RGB visualization
- Applies normalization & visualization
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# File Paths
file_path = "Data/Cuprite Nevada/ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.tif"
Meta_Data = "Data/Cuprite Nevada/ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-METADATA.xml"

def min_max_scale(band):
    """Applies Min-Max scaling (0-1) while avoiding division by zero."""
    min_val, max_val = np.min(band), np.max(band)
    if max_val - min_val == 0:
        return np.zeros_like(band)  # Prevents NaN if all values are the same
    return (band - min_val) / (max_val - min_val + 1e-6)  # Small epsilon for numerical stability

# --- Extract Wavelengths from XML ---
tree = ET.parse(Meta_Data)
root = tree.getroot()
wavelengths = []
band_map = {}
for band in root.findall(".//specific/bandCharacterisation/bandID"):
    band_id = int(band.attrib["number"]) # Extract band number
    wavelength = band.find("wavelengthCenterOfBand")
    
    if wavelength is not None:
        wl_value = float(wavelength.text.strip())
        wavelengths.append(wl_value)
        band_map[wl_value] = band_id  # Store mapping
    else:
        print("Warning: Missing wavelength value in metadata.")

wavelengths = np.array(wavelengths)  # Convert to NumPy array for processing

#print("Extracted Wavelengths:", wavelengths)

# --- Select RGB Bands Based on Wavelengths ---
target_wavelengths = [2199.45, 1653, 1047.84]  # Red, Green, Blue target wavelengths
selected_bands = []
for wl in target_wavelengths:
    if wl in band_map:
        selected_bands.append(band_map[wl])  # Exact match → Use bandID
    else:
        # Find the closest wavelength in band_map
        closest_wavelength = min(band_map.keys(), key=lambda x: abs(x - wl))
        selected_bands.append(band_map[closest_wavelength])  # Retrieve correct bandID


# Ensure valid band selection
r, g, b = selected_bands
print(f"Selected Bands: Red={r}, Green={g}, Blue={b}")

# --- Read HSI Data ---
with rasterio.open(file_path) as src:
    hsi = src.read().astype(np.float32)  # Convert to float to avoid int16 issues

print("HSI Shape:", hsi.shape)  # (Bands, Height, Width)

# --- Handle Negative Values Before Normalization ---
hsi[hsi < 0] = np.min(hsi[hsi >= 0])  # Replace negatives with the smallest positive value

# --- Normalize and Create RGB Image ---
rgb_image = np.dstack([min_max_scale(hsi[r]), 
                       min_max_scale(hsi[g]), 
                       min_max_scale(hsi[b])])

# --- Display the Image ---
plt.imshow(rgb_image)
plt.title("False Color Composite")
plt.axis("off")
plt.show()
