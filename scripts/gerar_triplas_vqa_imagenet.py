import os
import itertools
import numpy as np
import pandas as pd
import logging

DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join( os.path.join(os.path.join(DATA_DIR,"vqa"))  ,"mscoco")
IMAGENET_DIR = os.path.join( os.path.join(os.path.join(DATA_DIR,"imagenet"))  ,"images")


#################################################################
#               Configurando logs de execuçao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/train_siamese.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

logger.debug("DATA_DIR     : %s", DATA_DIR)
logger.debug("VQA_DIR      : %s", VQA_DIR)
logger.debug("IMAGENET_DIR : %s", IMAGENET_DIR)

i = 0
images_pairs = []
for vqa_img in os.listdir(VQA_DIR):    
    for imganet_img in os.listdir(IMAGENET_DIR):        
        images_pairs.append( [ vqa_img, imganet_img ] )
        if i % 100000 == 0:
            logger.debug("{:d} pares de imagens gerados".format(i))
    i = i + 1

logger.info("salvando arquivo de pares em {:s}".format(os.path.join(DATA_DIR, "image_pairs.csv")))
df = pd.DataFrame(images_pairs, columns=['mscoco','imagenet'])
df.to_csv(os.path.join(DATA_DIR, "image_pairs.csv"), index=0, header=0)

logger.info("Finalizado")


