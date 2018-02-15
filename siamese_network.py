#################################################################
#                         Rede Neural Simaesa                   #
#################################################################

import os
import sys
import itertools
import numpy as np
import pandas as pd
from random import shuffle

from scipy.misc import imresize

from keras.models import Model
from keras.applications import resnet50, xception

DATA_DIR="/media/ramon/dados/dataset/vqa/"
IMAGE_DIR= os.path.join(DATA_DIR, 'mscoco')

lista_triplas = os.path.join("data/", "triples_train.csv")
image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
image_triples = image_triples.values.tolist()

