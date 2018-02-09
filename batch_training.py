import os

import numpy as np
import pandas as pd


from numpy import genfromtxt


#################################################################
#                         CONSTANTES                            #
#################################################################

DATA_DIR = os.path.join(os.environ['DATA_DIR'], "vqa")
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

#################################################################
#                          Funções                              #
#################################################################
def criar_triplas(lista_imagens):        
    data = pd.read_csv(lista_imagens, sep=",", header=1, names=["image_id","category_id"])
    
    

#################################################################
#                       Inicio da Execucao                      #
#################################################################

lista_imagens = os.path.join(DATA_DIR, "train_ids.csv")
triplas = criar_triplas(lista_imagens)

print(triplas)