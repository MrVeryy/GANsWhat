import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image

"""
Interpolation Algorithms for Image Resize
=========================================

1. Nearest Neighbor (最近鄰)
   - 計算方式: 直接取最近的像素值
   - 特點: 最快，簡單
   - 優點: 運算量極小，速度快
   - 缺點: 鋸齒感重，細節遺失
   - 適用場景: Pixel Art 放大、需要保留「方格感」時

2. Bilinear (雙線性)
   - 計算方式: 在 X 與 Y 方向做線性插值 (2×2 鄰域)
   - 特點: 品質比 Nearest 好
   - 優點: 平滑過渡，比 Nearest 柔和
   - 缺點: 模糊，細節流失
   - 適用場景: 即時影像處理、一般用途

3. Bicubic (雙三次方)
   - 計算方式: 使用 4×4 鄰域 (16 像素)，做三次插值
   - 特點: 更平滑，細節較佳
   - 優點: 邊緣清晰，過渡自然
   - 缺點: 運算較慢，可能有些過度平滑
   - 適用場景: 攝影圖縮放、常見於 Photoshop

4. Lanczos
   - 計算方式: 使用 sinc 函數，取更大範圍 (常見 8×8)
   - 特點: 高品質、細節保留最好
   - 優點: 縮小圖效果極佳，保留高頻細節
   - 缺點: 計算最慢，邊緣可能出現振鈴 (ringing)
   - 適用場景: 高品質影像處理、出版、科研
"""

# ---------------------------------------------------------------------
# Load image
img = Image.open("./images/lenna.png").resize((512, 512))

# Define resize transforms
resize_nearest  = T.Resize((128, 128), interpolation=InterpolationMode.NEAREST)
resize_bilinear = T.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
resize_bicubic  = T.Resize((128, 128), interpolation=InterpolationMode.BICUBIC)
resize_lanczos  = T.Resize((128, 128), interpolation=InterpolationMode.LANCZOS)

imgs_resized = {
    "Nearest": resize_nearest(img),
    "Bilinear": resize_bilinear(img),
    "Bicubic": resize_bicubic(img),
    "Lanczos": resize_lanczos(img)
}

# ---------------------------------------------------------------------
# Create one figure with GridSpec
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3)  # 2 rows, 3 columns

# Left side: Original (span 2 rows, 1 column)
ax_orig = fig.add_subplot(gs[:, 0])  # span both rows
ax_orig.imshow(img)
ax_orig.set_title("Original (512x512)")
ax_orig.axis("off")

# Right side: 2x2 resized images
for ax, (name, im) in zip([fig.add_subplot(gs[0,1]), 
                           fig.add_subplot(gs[0,2]), 
                           fig.add_subplot(gs[1,1]), 
                           fig.add_subplot(gs[1,2])],
                          imgs_resized.items()):
    ax.imshow(im)
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()
