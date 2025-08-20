import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from mnist_data_downloader import download_minst_data
from mnist_models import Generator, Discriminator

# ---------------------
# 設定超參數
# ---------------------
latent_dim = 100
batch_size = 128
lr = 0.0002
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------
# 準備資料
# ---------------------
train_loader, _ = download_minst_data(batch_size=batch_size)

# ---------------------
# 建立模型
# ---------------------
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

# Loss function: Binary Cross Entropy
criterion = nn.BCELoss()

# Optimizers (Adam 常用於 GAN)
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ---------------------
# 訓練開始
# ---------------------
G_losses = []
D_losses = []

for epoch in range(epochs):
    for real_imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 標籤: 真=1, 假=0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # 1. 訓練 Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # 真圖片
        outputs_real = D(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        # 假圖片
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        outputs_fake = D(fake_imgs.detach())  # detach 避免更新 G
        d_loss_fake = criterion(outputs_fake, fake_labels)

        # 總 loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 2. 訓練 Generator
        # ---------------------
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        outputs = D(fake_imgs)

        # 目標: 讓 D 認為假的也是真的 (label=1)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # 紀錄 loss
        # ---------------------
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

    print(f"Epoch [{epoch+1}/{epochs}] | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
    
# ---------------------
# 儲存訓練好的 Generator
# ---------------------
import os
os.makedirs("./models", exist_ok=True)  # 如果資料夾不存在就建立
torch.save(G.state_dict(), "./models/generator_mnist.pth")
print("✅ Generator model saved at ./models/generator_mnist.pth")


# ---------------------
# 畫 loss 曲線
# ---------------------
plt.figure(figsize=(10,5))
plt.plot(G_losses, label="G Loss")
plt.plot(D_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ---------------------
# 生成一些假圖看看
# ---------------------
z = torch.randn(16, latent_dim).to(device)
fake_imgs = G(z).cpu().detach()

fig, axes = plt.subplots(4, 4, figsize=(6,6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fake_imgs[i][0], cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")
plt.show()
