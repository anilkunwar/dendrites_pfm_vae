import os
import io
import sys
import math
import types
import torch
import torch.nn as nn
import streamlit as st
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# ==========================================================
# 1. ARCHITECTURE DEFINITIONS (MODEL-SAFE)
# ==========================================================

def smooth_scale(x):
    return torch.sigmoid(x)

# ------------------------------
# Residual Blocks
# ------------------------------

class MultiKernelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        out = self.bn(out)
        out = out + self.shortcut(x)
        return self.relu(out)

# ------------------------------
# Encoder (FIXED)
# ------------------------------

class ResEncoder(nn.Module):
    def __init__(self, img_size, in_channels, base_channels, latent_size):
        super().__init__()

        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )

        # 🔑 FIX: infer conv_dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size[0], img_size[1])
            out = self.layers(dummy)
            conv_dim = out.flatten(1).shape[1]

        self.fc_mu = nn.Linear(conv_dim, latent_size)
        self.fc_logvar = nn.Linear(conv_dim, latent_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc_mu(x), self.fc_logvar(x)

# ------------------------------
# Decoder Blocks
# ------------------------------

class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.deconv(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.shortcut(x))

class GroupResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1,
                                         groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1,
                               groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.deconv(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.short(x))

class ChannelWiseBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class ResDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, latent_size, H, W):
        super().__init__()
        self.H, self.W = H, W
        init_h, init_w = H // 8, W // 8

        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)

        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),
            ResUpBlock(base_channels * 2, base_channels),
        )

        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels

        self.align_proj = nn.Conv2d(base_channels, self.group_channels, 1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        self.group_up = GroupResUpBlock(
            self.group_channels, self.group_channels, groups=out_channels
        )

        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        self.head = nn.Conv2d(self.group_channels, out_channels, 1,
                              groups=out_channels, bias=True)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, self.H // 8, self.W // 8)
        x = self.shared_up(x)
        x = self.align_bn(self.align_proj(x))
        x = self.group_up(x)
        x = self.refine(x)
        return smooth_scale(self.head(x))

# ------------------------------
# VAE
# ------------------------------

class VAE(nn.Module):
    def __init__(self, image_size=(3, 128, 128),
                 latent_size=64, hidden_dimension=64,
                 num_params=15, ctr_head_hidden=256):
        super().__init__()
        self.C, self.H, self.W = image_size

        self.encoder = ResEncoder(
            (self.H, self.W), self.C, hidden_dimension, latent_size
        )
        self.decoder = ResDecoder(
            self.C, hidden_dimension, latent_size, self.H, self.W
        )

        self.ctr_head = nn.Sequential(
            nn.Linear(latent_size, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, num_params),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        ctr_pred = self.ctr_head(z)
        return recon, mu, logvar, ctr_pred, z

# ==========================================================
# 2. DUMMY MODULE PATCH (UNPICKLING FIX)
# ==========================================================

m = types.ModuleType("src")
m.modelv9 = types.ModuleType("modelv9")

for cls in [
    VAE, ResEncoder, ResDecoder,
    MultiKernelResBlock, ResUpBlock,
    GroupResUpBlock, ChannelWiseBlock
]:
    setattr(m.modelv9, cls.__name__, cls)

sys.modules["src"] = m
sys.modules["src.modelv9"] = m.modelv9

# ==========================================================
# 3. STREAMLIT MODEL LOADING
# ==========================================================

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = ["knowledge_base", "knowledge-base", "."]

    for folder in folders:
        path = os.path.join(base_dir, folder)
        if os.path.exists(os.path.join(path, "vae_model.pt.part1")):
            parts = [os.path.join(path, f"vae_model.pt.part{i}") for i in range(1, 5)]
            buffer = io.BytesIO()
            for p in parts:
                with open(p, "rb") as f:
                    buffer.write(f.read())
            buffer.seek(0)
            model = torch.load(buffer, map_location="cpu", weights_only=False)
            model.eval()
            return model

    st.error("❌ Model parts not found")
    return None

# ==========================================================
# 4. STREAMLIT UI
# ==========================================================

st.title("VAE Image Reconstruction App")

model = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload an image", ["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    x = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])(img).unsqueeze(0)

    with torch.no_grad():
        recon, _, _, ctr, _ = model(x)

    st.image(
        transforms.ToPILImage()(recon.squeeze(0)),
        caption="Reconstruction",
        use_container_width=True
    )

    st.subheader("Predicted Control Parameters")
    st.table(ctr.squeeze(0).numpy())
