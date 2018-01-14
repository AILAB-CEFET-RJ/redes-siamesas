import os
import sys
from PIL import Image


DATA_DIR = "/Volumes/Externo/cefet/dataset/"
IMAGE_DIR = os.path.join(DATA_DIR, "imagenet")


for img_file in os.listdir(IMAGE_DIR):
    try:
        if(img_file != ".DS_Store"):
            im = Image.open(os.path.join(IMAGE_DIR, img_file))
    except:
        os.remove(os.path.join(IMAGE_DIR, img_file))
        print(img_file, "apagado")