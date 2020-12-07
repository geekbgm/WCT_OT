import pytorch_ssim
import torch
from torch.autograd import Variable
import cv2
import numpy as np

content = cv2.imread("./examples/content/in00.png")
result = cv2.imread("./outputs/in00_cat5_decoder_encoder_skip..png")

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))