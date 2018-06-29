import os
import sys
from PIL import Image

DATA_DIR = os.environ["DATA_DIR"]
IMAGENET_DIR = os.path.join(DATA_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")
DESTINATION_DIR = os.path.join(DATA_DIR, "upload")

SIZE = 299, 299

def varrer_diretorios(imagenet_dir):    
    i = 0
    for synset_dir in os.listdir(imagenet_dir):
        for imagenet_file in os.listdir(os.path.join(IMAGENET_DIR, synset_dir)):
           try:
                im = Image.open( os.path.join(IMAGENET_DIR, synset_dir, imagenet_file))
                im.thumbnail(SIZE, Image.ANTIALIAS)
                im.convert("RGB").save(os.path.join(DESTINATION_DIR, imagenet_file))
                i = i + 1
                if( i % 1000 == 0):
                    print(i, "imagens movidas")
           except Exception as err:
                print(err)
                print("Erro ao processar " + imagenet_file)
                sys.exit()
    
    print("pronto", i, "imagens movidas")


varrer_diretorios(IMAGENET_DIR)


print("Finalizado")