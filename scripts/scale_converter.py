import numpy as np
import os
import sys
import cv2
from PIL import Image


"""DATA_DIR = "/home/rsilva/datasets/vqa/"
IMAGE_DIR = os.path.join(DATA_DIR, "val2014")
NEW_IMAGE_DIR = os.path.join(DATA_DIR, "mscoco_val")
"""

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR, "imagenet", "images")
NEW_IMAGE_DIR = os.path.join(DATA_DIR, "imagenet", "convertido")

i = 0
for img_file in os.listdir(IMAGE_DIR):        
    try:
        im = Image.open( os.path.join(IMAGE_DIR,img_file))
        im.convert('RGB').save(os.path.join(NEW_IMAGE_DIR, img_file))
        i = i + 1
        if( i % 1000 == 0):
            print(i, "imagens convertidas")
    except:
        print("Erro ao processar " + img_file)