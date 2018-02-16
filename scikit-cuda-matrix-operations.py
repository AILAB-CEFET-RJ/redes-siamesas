import os
import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as linalg
from sklearn.model_selection import train_test_split, KFold

linalg.init()

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict


def preprocessar_dados(vec_dict, triplas, train_size=0.7):
    xdata, ydata = [], []
    
    for image_triple in triplas:
        X1 = vec_dict[image_triple[0]]
        X2 = vec_dict[image_triple[1]]
        print(X1)
        return
        #xdata.append(np.power(np.subtract(X1, X2), 2))
        #ydata.append(image_triple[2])
    #X, y = np.array(xdata), np.array(ydata)
    #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
   
    #return Xtrain, Xtest, ytrain, ytest

DATA_DIR = os.environ["DATA_DIR"]
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")

amostra = os.path.join(DATA_DIR, 'train_2014_50.csv')

vec_dict = carregar_vetores(VECTOR_FILE)
preprocessar_dados(vec_dict, amostra)