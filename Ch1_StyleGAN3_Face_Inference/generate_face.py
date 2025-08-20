import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

import torch
import numpy as np

from PIL import Image
from stylegan_utils import dnnlib
from stylegan_utils import legacy

# -----------------------------------------------------------------------
# Packages Check
print("=== Package Check ===")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("numpy version:", np.__version__)
print("PIL version:", Image.__version__)

# dnnlib / legacy 沒有 __version__，就確認模組名稱
print("dnnlib imported from:", dnnlib.__file__)
print("legacy imported from:", legacy.__file__)
print("All imports OK ✔")
# -----------------------------------------------------------------------
# Seed
seed = 133

# Model path 
model = "models/stylegan3-t-ffhq-1024x1024.pkl"

# Device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Using device: ", device)

# load the Generator
with open(model, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# latent vector Z
rnd = np.random.RandomState(seed)
z = torch.from_numpy(rnd.randn(1, G.z_dim)).to(device)

# Generate image
img = G(z, None, truncation_psi=0.7, noise_mode='const')  
img = (img.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8) # NCHW, [-1,1]
img = img[0].permute(1, 2, 0).cpu().numpy()  # HWC

out_path = f"output/{int(time.time())}_face_seed_{seed}.png"
Image.fromarray(img, 'RGB').save(out_path)
print("Saved:", out_path)






