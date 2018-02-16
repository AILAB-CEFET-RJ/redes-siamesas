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
from sklearn.model_selection import StratifiedKFold

seed = 7

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict


DATA_DIR="/media/ramon/dados/dataset/vqa/"
IMAGE_DIR= os.path.join(DATA_DIR, 'mscoco')
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")

print("Carregando as triplas de imagens")

lista_triplas = os.path.join("data/", "triples_train.csv")
image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
image_triples = image_triples.values.tolist()

print("Carregando os vetores de imagens")
vec_dict = carregar_vetores(VECTOR_FILE)

X = image_triples[:,0:2]
Y = image_triples[:,2]

kfold = StratifiedKFold(n_splits=100, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(X, Y):
    X1 = vec_dict[X[0]]
    X2 = vec_dict[X[1]]