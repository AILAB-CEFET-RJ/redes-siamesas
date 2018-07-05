import os, sys
import gensim
import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

DATA_DIR = os.environ["DATA_DIR"]
VECTORS_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
WORDS_FILE = os.path.join(DATA_DIR, 'questions-words.txt')

DISTANCES_FOLDER = os.path.join(DATA_DIR, 'distances')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)
model.accuracy(WORDS_FILE)

labels = pd.read_csv(os.path.join(DATA_DIR, "imagenet_labels.csv"), names=['imagenet_img_id', 'label'], header=0)
nouns = pd.read_csv(os.path.join(DATA_DIR, "vqa_words.csv"), names=['vqa_img_id', 'imagenet_img_id', 'noun'], header=0)

print("Processando distancias")

nouns_vectors = []

for index, row in nouns.iterrows():
    if row["noun"] in model:        
        nouns_vectors.append( model[row["noun"]] )

labels_vectors = []
for k, v in labels.iterrows():    
    if v["label"] in model:
        labels_vectors.append( model[v["label"]] )
		
distances = euclidean_distances(nouns_vectors, labels_vectors)

print("Liberando memoria...")
del nouns_vectors[:]
del labels_vectors[:]

tam_i, tam_j = distances.shape
	  
for i in range(0, tam_i - 1):
	  for j in range(0, tam_j - 1):
	  	print( nouns.loc[i:0], nouns.loc[i:1], labels.loc[i:0], labels.loc[i:0], distances[i,j] )
	  	sys.exit()
	 
df_dist = pd.DataFrame(distances.flatten(), columns=['distance'])
plt.figure(figsize=(12,6))

df_dist['distance'].hist(log=True, grid=False)
plt.xlabel("Distance")
plt.title("Distances")
plt.show()