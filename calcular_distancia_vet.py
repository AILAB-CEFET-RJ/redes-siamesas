import os
import sys
import numpy as np
import pandas as pd
from random import shuffle
from utils import calc, dados
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

def preprocessar_dados(vec_dict, triplas, train_size=0.7):
    dist = euclidean_distances(vec_dict, vec_dict)
    return dist

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

print("Carregando vetores")
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
vec_dict = carregar_vetores(VECTOR_FILE)

print("Carregando triplas")
TRIPLES_FILES = os.path.join(DATA_DIR, "triples_train_50.csv")
image_triples = dados.carregar_triplas(TRIPLES_FILES)

print("Calculando distancia")
data = preprocessar_dados(vec_dict, image_triples)

df = pd.DataFrame(data, columns=['distance', 'similar'])
print(df.head(25))
df.to_csv(os.path.join(DATA_DIR, "distances_50.csv"))

print("Salvo")
print("Finalizado")