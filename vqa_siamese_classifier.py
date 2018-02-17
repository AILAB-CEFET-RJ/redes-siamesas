#################################################################
#                         Rede Neural Siamesa                   #
#################################################################

import os
import sys
import itertools
import numpy as np
import pandas as pd
from random import shuffle
import logging


from utils import dados
from utils import neuralnetwork as nn

seed = 7
np.random.seed(seed)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',                    
                    filemode='w')

DATA_DIR="/media/ramon/dados/dataset/vqa/"
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
TRIPLES_FILES = os.path.join("data/", "triples_train.csv")
BATCH_SIZE = 32
NUM_EPOCHS = 10
VECTOR_SIZE = 2048

logging.debug("DATA_DIR %s", DATA_DIR)
logging.debug("VECTOR_FILE %s", VECTOR_FILE)
logging.debug("TRIPLES_FILES %s", TRIPLES_FILES)
logging.debug("BATCH_SIZE %s", BATCH_SIZE)
logging.debug("NUM_EPOCHS %s", NUM_EPOCHS)
logging.debug("VECTOR_SIZE %s", NUM_EPOCHS)

logging.info("Carregando as triplas de imagens...")
image_triples = dados.carregar_triplas(TRIPLES_FILES)

logging.info("Carregando os vetores de imagens...")
vec_dict = dados.carregar_vetores(VECTOR_FILE)

#X = image_triples[...,0:2]
#Y = image_triples[:,2].astype(int)

train_triples, val_triples, test_triples = dados.train_test_split(image_triples, splits=[0.7,0.1,0.2])

logging.info("train_triples %d", len(train_triples))
logging.info("val_triples %d", len(val_triples))
logging.info("test_triples %d", len(test_triples))

train_gen = dados.gerador_de_lotes(train_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)
val_gen = dados.gerador_de_lotes(val_triples, VECTOR_SIZE, vec_dict, BATCH_SIZE)

model = nn.base_model(VECTOR_SIZE)

logging.info("Finalizado")