import os
import numpy as np
import pandas as pd

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

def carregar_triplas(lista_triplas):
    image_triples = pd.read_csv(lista_triplas, sep=",", header=0, names=["left","right","similar"])
    return image_triples.values

def train_test_split(triples, splits):
    assert sum(splits) == 1.0
    split_pts = np.cumsum(np.array([0.] + splits))
    indices = np.random.permutation(np.arange(len(triples)))
    shuffled_triples = [triples[i] for i in indices]
    data_splits = []
    for sid in range(len(splits)):
        start = int(split_pts[sid] * len(triples))
        end = int(split_pts[sid + 1] * len(triples))
        data_splits.append(shuffled_triples[start:end])
    return data_splits

def gerador_de_lotes(triplas, tam_vetor, vetor, tamanho_lote=32):
    while True:
        # shuffle once per batch
        indices = np.random.permutation(np.arange(len(triplas)))
        num_batches = len(triplas) // tam_vetor
        for bid in range(num_batches):
            batch_indices = indices[bid * tam_vetor : (bid + 1) * tam_vetor]
            batch = [triplas[i] for i in batch_indices]
            yield lotear_vetores(batch, tam_vetor, vetor)

def lotear_vetores(lote, tam_vetor, vetor):
    X1 = np.zeros((len(lote), tam_vetor))
    X2 = np.zeros((len(lote), tam_vetor))
    Y = np.zeros((len(lote), 2))
    for tid in range(len(lote)):
        X1[tid] = vetor[lote[tid][0]]
        X2[tid] = vetor[lote[tid][1]]
        Y[tid] = [1, 0] if lote[tid][2] == 0 else [0, 1]
    return ([X1, X2], Y)