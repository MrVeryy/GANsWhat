import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms as T
from stylegan_utils import dnnlib
from stylegan_utils import legacy

# ------------------------------------------------------------------
# models
model = "./models/karras2019stylegan-ffhq-1024x1024.pkl"

# device
# device = global_utils.get_recom_device()
device = torch.device('mps')
print("use devices: ", device)

# ------------------------------------------------------------------
# reference: https://github.com/abhijitpal1247/Image2StyleGAN/blob/master/project.py














