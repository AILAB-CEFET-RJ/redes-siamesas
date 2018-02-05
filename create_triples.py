import os

import numpy as np
import itertools
import pandas as pd
from sklearn.utils import shuffle

np.random.seed(7)

DATA_DIR = "data/"
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")
MODEL_DIR = os.path.join(DATA_DIR,"models")


def criar_triplas(image_dir, lista_imagens):
    data = pd.read_csv(lista_imagens, sep=",", header=1, names=["image_id","filename","category_id"])
    image_cache = {}
    for index, row in data.iterrows():
        id = row["filename"]
        if(id in image_cache):
            image_cache[id]["categories"].append(row["category_id"])
        else:
            image_cache[id] = {"image_id" : id, "filename" : row["filename"], "categories" : [row["category_id"]]}
    #Triplas que serao retornadas
    triplas = []

    for index, row in image_cache.items():
        for _i, _r in image_cache.items():
            if index != _i:
                _match = set(row["categories"]).intersection(_r["categories"])
                if len(_match) > 0:
                    triplas.append((row["filename"], _r["filename"], 1))
                else:
                    triplas.append((row["filename"], _r["filename"], 0))
    return shuffle(triplas)


def salvar_triplas(triplas):
    labels = ['image_1', 'image_2', 'similarity']
    df = pd.DataFrame.from_records(image_triples, columns=labels)
    df.to_csv(DATA_DIR, "triplas.csv")



#################################################################
#                       Inicio da Execucao                      #
#################################################################


lista_imagens = os.path.join(DATA_DIR, 'train_10.csv')
print("Criando triplas")
image_triples = criar_triplas(IMAGE_DIR, lista_imagens)
print("Pronto !!!")

salvar_triplas(image_triples)
print("Salvo")