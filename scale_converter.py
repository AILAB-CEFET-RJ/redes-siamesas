import numpy as np
import os
import sys
import cv2
from PIL import Image


DATA_DIR = "/home/rsilva/Projects/cefet/dataset"
IMAGE_DIR = os.path.join(DATA_DIR, "imagenet")
NEW_IMAGE_DIR = os.path.join(DATA_DIR, "convertido")


for img_file in os.listdir(IMAGE_DIR):        
    im = Image.open( os.path.join(IMAGE_DIR,img_file))
    if(len(im.shape)<3):
        try:
            print(img_file, "is greyscale")
            im.convert('RGB').save(img_file)
        except:
            print("Deu merda")
