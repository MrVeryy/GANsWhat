import torch
import matplotlib.pyplot as plt
from mnist_models import Generator

latent_dim = 100

# 載入訓練好的 Generator
G = Generator(latent_dim=latent_dim)
G.load_state_dict(torch.load("./models/generator_mnist.pth", map_location="cpu"))
G.eval()

# 生成 16 張隨機手寫數字
z = torch.randn(16, latent_dim)
fake_imgs = G(z).detach()

# 畫圖
fig, axes = plt.subplots(4, 4, figsize=(6,6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_imgs[i][0], cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")
plt.show()
