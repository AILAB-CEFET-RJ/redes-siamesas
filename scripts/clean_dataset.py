import os
import sys
from PIL import Image
import hashlib

DATA_DIR = os.environ["DATA_DIR"]
IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet")
IMAGE_DIR = os.path.join(IMAGENET_DIR, "images")

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

for img_file in os.listdir(IMAGE_DIR):
    try:
        if(img_file != ".DS_Store"):
            im = Image.open(os.path.join(IMAGE_DIR, img_file))

        if( md5(os.path.join(IMAGE_DIR, img_file)) == "880a7a58e05d3e83797f27573bb6d35c" ):
            im.close()
            os.remove(os.path.join(IMAGE_DIR, img_file))
            print(img_file, "apagado")
    except:
        os.remove(os.path.join(IMAGE_DIR, img_file))
        print(img_file, "apagado")

print("Finalizado")