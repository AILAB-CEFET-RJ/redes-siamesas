import os
import itertools
import numpy as np
import pandas as pd
import logging

DATA_DIR = os.environ["DATA_DIR"]
VQA_DIR = os.path.join( os.path.join(os.path.join(DATA_DIR,"vqa"))  ,"mscoco")
IMAGENET_DIR = os.path.join( os.path.join(os.path.join(DATA_DIR,"imagenet"))  ,"images")
PAIRS_DIR =  os.path.join(os.path.join(DATA_DIR,"triplas"))

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
logger.debug("PAIRS_DIR    : %s", PAIRS_DIR)


i = 0
images_pairs = []
for vqa_img in os.listdir(VQA_DIR):    
    for imagenet_img in os.listdir(IMAGENET_DIR):     
        images_pairs.append( [ vqa_img, imagenet_img ] )
   
    imagene_name = "{:s}_pairs.csv".format(vqa_img)
    df = pd.DataFrame(images_pairs, columns=['mscoco','imagenet'])
    df.to_csv(os.path.join(PAIRS_DIR, imagene_name), index=0, header=0)
    
    if i % 1000 == 0:
        logger.debug("{:d} pares de imagens gerados".format(i))
    images_pairs = []
    i = i + 1

logger.info("Finalizado")


