import torch
import torch.nn as nn

# -----------------------------------------------------------------
# Generator (G)
# -----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.output_dim = int(torch.prod(torch.tensor(img_shape)))  # 1*28*28=784

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, self.output_dim),
            nn.Tanh()   # 輸出範圍 [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *self.img_shape)  # reshape 回 [batch, 1, 28, 28]
        return img


# -----------------------------------------------------------------
# Discriminator (D)
# -----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.input_dim = int(torch.prod(torch.tensor(img_shape)))  # 784

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()  # 輸出 [0,1]，代表真實概率
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)  # 攤平成向量
        validity = self.model(flat)
        return validity
