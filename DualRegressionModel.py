import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
torch.autograd.set_detect_anomaly(True)

# ------------------------
# 1️ Load Hyperspectral Data
# ------------------------
def load_hsi(filepath):
    with rasterio.open(filepath) as src:
        hsi_data = src.read().astype(np.float32)  # Shape: (Bands, Height, Width)
    return torch.tensor(hsi_data)  # Convert to Tensor

file_path = "Data/Cuprite Nevada/ENMAP01-____L2A-DT0000025905_20230707T192008Z_001_V010303_20230922T131734Z-SPECTRAL_IMAGE.tif"
hsi_tensor = load_hsi(file_path)  # (Bands, Height, Width)

# Normalize data
hsi_tensor = (hsi_tensor - hsi_tensor.min()) / (hsi_tensor.max() - hsi_tensor.min())

# Add batch dimension and downsample
if hsi_tensor.ndim == 3:
    hsi_tensor = hsi_tensor.unsqueeze(0)  # (1, Bands, Height, Width)
hsi_tensor = F.interpolate(hsi_tensor, scale_factor=0.1, mode='bilinear', align_corners=False)  # Reduce size 10x to avoid memory errors

# Move spectral bands to channel dimension
hsi_tensor = hsi_tensor.permute(0, 2, 3, 1)  # (1, Height, Width, Bands)

# Add noise
def add_simulated_noise(hsi):
    gauss_noise = torch.randn_like(hsi) * 0.05
    poisson_noise = torch.poisson(hsi * 10) / 10 - hsi
    return hsi + gauss_noise + poisson_noise

noisy_hsi_tensor = add_simulated_noise(hsi_tensor)

# Move back to (Batch, Channels, Height, Width)
hsi_tensor = hsi_tensor.permute(0, 3, 1, 2)
noisy_hsi_tensor = noisy_hsi_tensor.permute(0, 3, 1, 2)

# ------------------------
# 3️ Define Model Architecture: CNN + Transformer
# ------------------------
class SpectralAttention(nn.Module):
    def __init__(self, num_bands):
        super(SpectralAttention, self).__init__()
        self.embed_dim = num_bands
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=4, batch_first=True)  

    def forward(self, x):
        batch_size, num_bands, height, width = x.shape
        x = F.adaptive_avg_pool2d(x, (32, 32))  # Reduce spatial size
        x = x.view(batch_size, num_bands, -1).permute(0, 2, 1)  # (Batch, Seq, Features)

        attn_output, _ = self.attention(x, x, x)  # Ensure correct input

        seq_len, num_bands = attn_output.shape[1], attn_output.shape[2]
        sqrt_len = int(seq_len ** 0.5)

        if sqrt_len * sqrt_len != seq_len:
            raise ValueError(f"Expected square-shaped spatial output, but got seq_len={seq_len}")

        attn_output = attn_output.permute(0, 2, 1).reshape(batch_size, num_bands, sqrt_len, sqrt_len)

        return attn_output

class Denoiser(nn.Module):
    def __init__(self, num_bands):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.attn = SpectralAttention(64)
        self.conv2 = nn.Conv2d(64, num_bands, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.attn(x)
        x = self.conv2(x)
        return x

class NoiseAdder(nn.Module):
    def __init__(self, num_bands):
        super(NoiseAdder, self).__init__()
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, num_bands, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.conv2(x)
        return x

# ------------------------
# 4️ Spectral Angle Mapper (SAM) Loss
# ------------------------
def spectral_angle_loss(denoised, original):
    dot_product = torch.sum(original * denoised, dim=1)
    norm_orig = torch.norm(original, dim=1)
    norm_denoised = torch.norm(denoised, dim=1)
    cos_theta = dot_product / (norm_orig * norm_denoised + 1e-6)
    sam_loss = torch.acos(torch.clamp(cos_theta, -1, 1)).mean()
    return sam_loss

# ------------------------
# 5️ Initialize Models, Optimizers
# ------------------------
num_bands = hsi_tensor.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

denoiser = Denoiser(num_bands).to(device)
noise_adder = NoiseAdder(num_bands).to(device)

optimizer_denoiser = optim.Adam(denoiser.parameters(), lr=0.001)
optimizer_noise_adder = optim.Adam(noise_adder.parameters(), lr=0.001)

mse_loss = nn.MSELoss()

# ------------------------
# 6️ Training Loop
# ------------------------
epochs = 50

hsi_tensor = hsi_tensor.to(device)
noisy_hsi_tensor = noisy_hsi_tensor.to(device)

for epoch in range(epochs):
    # ------------------- Denoiser Step -------------------
    optimizer_denoiser.zero_grad()

    denoised = denoiser(noisy_hsi_tensor)

    # Resize denoised tensor to match hsi_tensor dimensions
    denoised_resized = F.interpolate(denoised, size=(hsi_tensor.shape[2], hsi_tensor.shape[3]), mode="bilinear", align_corners=False)

    # Compute loss
    loss_denoiser = mse_loss(denoised_resized, hsi_tensor) + spectral_angle_loss(denoised_resized, hsi_tensor)
    loss_denoiser.backward(retain_graph=True)
    optimizer_denoiser.step()

    # ------------------- Noise Adder Step -------------------
    optimizer_noise_adder.zero_grad()

    generated_noise = noise_adder(denoised.detach())  # Detach to prevent modifying denoiser gradients
    noisy_reconstructed = denoised.detach() + generated_noise

    # Resize to match original noisy input
    noisy_reconstructed_resized = F.interpolate(noisy_reconstructed, size=(noisy_hsi_tensor.shape[2], noisy_hsi_tensor.shape[3]), mode="bilinear", align_corners=False)

    # Compute loss
    loss_noise_adder = mse_loss(noisy_reconstructed_resized, noisy_hsi_tensor)
    loss_noise_adder.backward()
    optimizer_noise_adder.step()

    # ------------------- Logging -------------------
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | Denoiser Loss: {loss_denoiser.item():.6f} | Noise Adder Loss: {loss_noise_adder.item():.6f}")


   
# ------------------------
# 7️ Visualize Results
# ------------------------
denoised = denoiser(noisy_hsi_tensor).squeeze(0).cpu().detach().numpy()
# Convert denoised NumPy array back to PyTorch tensor before passing to Noise Adder
denoised_tensor = torch.tensor(denoised).unsqueeze(0).to(device)

# Get the reconstructed noisy image from the Noise Adder
reconstructed_noisy_tensor = denoised_tensor + noise_adder(denoised_tensor)

# Convert back to NumPy for visualization
reconstructed_noisy = reconstructed_noisy_tensor.squeeze(0).cpu().detach().numpy()

noisy = noisy_hsi_tensor.squeeze(0).cpu().detach().numpy()
original = hsi_tensor.squeeze(0).cpu().detach().numpy()

band_idx = 10  # Choose a spectral band to visualize

plt.figure(figsize=(20, 5))

# Original Noisy Image
plt.subplot(1, 4, 1)
plt.imshow(original[band_idx], cmap="gray")
plt.title("Original Noisy Image")

# Noisy Image (Before Denoising) [This is redundant]
plt.subplot(1, 4, 2)
plt.imshow(noisy[band_idx], cmap="gray")
plt.title("Noisy Image (Input)")

# Denoised Image
plt.subplot(1, 4, 3)
plt.imshow(denoised[band_idx], cmap="gray")
plt.title("Denoised Image")

# Reconstructed Noisy Image
plt.subplot(1, 4, 4)
plt.imshow(reconstructed_noisy[band_idx], cmap="gray")
plt.title("Reconstructed Noisy Image (Output of Noise Adder)")

plt.tight_layout()
plt.show()
