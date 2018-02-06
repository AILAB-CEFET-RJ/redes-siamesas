import os

import numpy as np
import pandas as pd


#################################################################
#                         CONSTANTES                            #
#################################################################

DATA_DIR = os.path.join(os.environ['DATA_DIR'], "vqa")
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

#################################################################
#                          Funções                              #
#################################################################
@profile
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

#################################################################
#                       Inicio da Execucao                      #
#################################################################

lista_imagens = os.path.join(DATA_DIR, "train_2014.csv")
criar_triplas(IMAGE_DIR, lista_imagens)