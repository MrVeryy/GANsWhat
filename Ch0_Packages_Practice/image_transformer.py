import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# ======================================================================
# Open image
img = Image.open("./images/lenna.png")

# ======================================================================
# Practice 1: Tensor <-> Image conversion
to_tensor = T.ToTensor()
img_tensor = to_tensor(img)
print(f'img -> T.ToTensor(): {type(img_tensor)}, shape={img_tensor.shape}')

to_image = T.ToPILImage()
print(f'img_tensor -> T.ToPILImage(): {type(to_image(img_tensor))}')

# ======================================================================
# Practice 2: Understanding Image Structure
print(f'Shape: {img_tensor.shape}')  # (C, H, W)

# ======================================================================
# Practice 3: Flip transforms
# RandomHorizontalFlip, RandomVerticalFlip
flip_h = T.RandomHorizontalFlip(p=1.0)  # always flip
flip_v = T.RandomVerticalFlip(p=1.0)    # always flip

img_flip_h = flip_h(img)
img_flip_v = flip_v(img)

img_flip_h.show(title="Horizontal Flip")
img_flip_v.show(title="Vertical Flip")

# ======================================================================
# Practice 4: Compose multiple transforms
compose_transform = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(p=1.0),
    T.ToTensor()
])

img_composed = compose_transform(img)
print(f'Compose result: {type(img_composed)}, shape={img_composed.shape}')

# ======================================================================
# Practice 5: Normalize
# Convert to tensor first
to_tensor = T.ToTensor()
img_tensor = to_tensor(img)

# Normalize: shift & scale RGB channels
normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
img_normalized = normalize(img_tensor)
print(f'After Normalize: min={img_normalized.min()}, max={img_normalized.max()}')

# ======================================================================
# Practice 6: Resize & CenterCrop
resize = T.Resize((100, 200))
center_crop = T.CenterCrop(100)

img_resized = resize(img)
img_cropped = center_crop(img_resized)

print(f'Resize -> {img_resized.size}, CenterCrop -> {img_cropped.size}')

# ======================================================================
# Practice 7: Convert Color
grayscale = T.Grayscale(num_output_channels=1)
img_gray = grayscale(img)
img_gray.show(title="Grayscale")
