import os
import sys
import types
import math
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ==========================================================
# 1. ARCHITECTURE DEFINITIONS (must match exactly what was saved)
# ==========================================================
def smooth_scale(x):
    return torch.sigmoid(x)


class MultiKernelResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
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


class ResEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )
        # We no longer hard-code conv_dim → we create Linear layers lazily
        self.fc_mu = None
        self.fc_logvar = None
        self._feature_dim = None
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.layers(x)
        flat = torch.flatten(x, 1)  # [B, C*H*W]

        if self.fc_mu is None:
            # First forward → initialize linear layers
            self._feature_dim = flat.shape[1]
            device = flat.device
            self.fc_mu = nn.Linear(self._feature_dim, self.latent_dim).to(device)
            self.fc_logvar = nn.Linear(self._feature_dim, self.latent_dim).to(device)
            # Optional: print for debugging (comment out in production)
            print(f"[Encoder] Auto-detected flattened dim: {self._feature_dim}")

        return self.fc_mu(flat), self.fc_logvar(flat)


class ResUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.deconv1(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.shortcut(x))


class GroupResUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.deconv(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.short(x))


class ChannelWiseBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
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
    def __init__(self, out_channels: int, base_channels: int, latent_dim: int, img_h: int, img_w: int):
        super().__init__()
        self.H, self.W = img_h, img_w
        init_h, init_w = img_h // 8, img_w // 8

        self.fc = nn.Linear(latent_dim, base_channels * 4 * init_h * init_w)

        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),
            ResUpBlock(base_channels * 2, base_channels),
        )

        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels

        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)

        self.group_up = GroupResUpBlock(self.group_channels, self.group_channels, groups=out_channels)

        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )

        self.head = nn.Conv2d(self.group_channels, out_channels, 1, groups=out_channels, bias=True)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, self.H // 8, self.W // 8)
        x = self.shared_up(x)
        x = self.align_bn(self.align_proj(x))
        x = self.group_up(x)
        x = self.refine(x)
        return smooth_scale(self.head(x))


class VAE(nn.Module):
    def __init__(self,
                 image_size=(3, 128, 128),
                 latent_dim=64,
                 hidden_channels=64,
                 num_control_params=15,
                 control_head_hidden=256):
        super().__init__()
        C, H, W = image_size
        self.image_size = (C, H, W)
        self.latent_dim = latent_dim

        self.encoder = ResEncoder(in_channels=C, base_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = ResDecoder(out_channels=C, base_channels=hidden_channels,
                                  latent_dim=latent_dim, img_h=H, img_w=W)

        self.control_head = nn.Sequential(
            nn.Linear(latent_dim, control_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(control_head_hidden, control_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(control_head_hidden, num_control_params),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        ctr_pred = self.control_head(z)
        return recon, mu, logvar, ctr_pred, z


# ==========================================================
# 2. Fake module structure so old pickles can load
# ==========================================================
m = types.ModuleType("src")
m.modelv9 = types.ModuleType("modelv9")
m.modelv9.VAE = VAE
m.modelv9.ResEncoder = ResEncoder
m.modelv9.ResDecoder = ResDecoder
m.modelv9.MultiKernelResBlock = MultiKernelResBlock
m.modelv9.ResUpBlock = ResUpBlock
m.modelv9.GroupResUpBlock = GroupResUpBlock
m.modelv9.ChannelWiseBlock = ChannelWiseBlock

sys.modules["src"] = m
sys.modules["src.modelv9"] = m.modelv9


# ==========================================================
# 3. Model loading with better diagnostics
# ==========================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    script_dir = Path(__file__).parent.resolve()

    possible_dirs = [
        script_dir / "knowledge_base",
        script_dir / "knowledge-base",
        script_dir / "model",
        script_dir,
        Path.cwd() / "knowledge_base",
        Path.cwd() / "knowledge-base",
    ]

    model_parts = None
    base_folder = None

    for d in possible_dirs:
        if not d.is_dir():
            continue
        parts = list(d.glob("vae_model.pt.part*"))
        if len(parts) >= 4:  # at least 4 parts expected
            model_parts = sorted(parts, key=lambda p: int(p.suffix[5:]))
            base_folder = d
            break

    if model_parts is None:
        st.error("Model parts not found.\nExpected: vae_model.pt.part1 ... vae_model.pt.part4\n"
                 "in folder: knowledge_base / knowledge-base / current directory")
        st.info(f"Searched locations:\n" + "\n".join(str(p) for p in possible_dirs))
        return None

    try:
        buffer = io.BytesIO()
        total_size = 0
        for part_path in model_parts:
            size = part_path.stat().st_size
            total_size += size
            with open(part_path, "rb") as f:
                buffer.write(f.read())
            st.caption(f"Loaded {part_path.name} ({size / 1024**2:.1f} MB)")

        buffer.seek(0)
        st.caption(f"Total merged size: {total_size / 1024**2:.1f} MB")

        # Important: weights_only=False because we load full model object (not state_dict)
        model = torch.load(buffer, map_location="cpu", weights_only=False)
        model.eval()

        # Quick smoke test
        try:
            dummy = torch.randn(1, 3, 128, 128)
            _ = model(dummy)
            st.success("Model loaded and test forward pass successful")
        except Exception as e:
            st.warning(f"Test forward pass failed — model might expect different input size\n{e}")

        return model

    except Exception as e:
        st.error(f"Failed to load model:\n{str(e)}")
        return None


# ==========================================================
# 4. Streamlit App
# ==========================================================
st.title("VAE – Reconstruction & Control Parameter Prediction")
st.markdown("Upload an image → get reconstruction + predicted control parameters")

model = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)

        # ────────────────────────────────────────────────
        # You can experiment here with different resolutions
        # Most common alternatives: (256,256), (192,192), (96,96)
        # ────────────────────────────────────────────────
        TARGET_SIZE = (128, 128)

        transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
        ])

        x = transform(image).unsqueeze(0)  # [1,3,H,W]

        with torch.no_grad():
            recon, mu, logvar, ctr_pred, z = model(x)

        # Visualize reconstruction
        recon_img = recon.squeeze(0).clamp(0, 1)
        recon_pil = transforms.ToPILImage()(recon_img)
        st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)

        # Show predictions
        st.subheader("Predicted Control Parameters")
        params = ctr_pred.squeeze(0).cpu().numpy()
        
        # Nice table with index
        data = {"Parameter": [f"p{i+1}" for i in range(len(params))],
                "Value": [f"{v:.4f}" for v in params]}
        st.dataframe(data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error during inference:\n{str(e)}")
        st.info("Possible causes:\n• Wrong image size expected by model\n• Corrupted model file\n• Architecture mismatch")
