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
    

#################################################################
#                       Inicio da Execucao                      #
#################################################################

lista_imagens = os.path.join(DATA_DIR, "train_2014.csv")
triplas = criar_triplas(IMAGE_DIR, lista_imagens)