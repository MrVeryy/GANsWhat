import torch
from mnist_models import Generator, Discriminator

latent_dim = 100
batch_size = 16

# 建立模型
G = Generator(latent_dim=latent_dim)
D = Discriminator()

# 測試生成假圖
z = torch.randn(batch_size, latent_dim)  # 隨機噪聲
fake_imgs = G(z)
print("Fake images shape:", fake_imgs.shape)  # [16, 1, 28, 28]

# 測試判別器
out = D(fake_imgs)
print("Discriminator output:", out.shape)  # [16, 1]
