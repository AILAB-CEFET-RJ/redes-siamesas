import numpy as np
import os
import sys
import cv2
from PIL import Image


DATA_DIR = "/home/rsilva/datasets/vqa/"
IMAGE_DIR = os.path.join(DATA_DIR, "train2014")
NEW_IMAGE_DIR = os.path.join(DATA_DIR, "mscoco")

i = 0
for img_file in os.listdir(IMAGE_DIR):        
    im = Image.open( os.path.join(IMAGE_DIR,img_file))
    im.convert('RGB').save(os.path.join(NEW_IMAGE_DIR, img_file))
    i = i +1
    if( i % 1000 == 0):
        print(i, "imagens convertidas")
