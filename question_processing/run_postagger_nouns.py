import json, os, sys, nltk
import pandas as pd
import csv
from nltk.tokenize import word_tokenize
import gensim
import logging
import pprint


DATA_DIR = os.environ["DATA_DIR"]
VECTORS_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
WORDS_FILE = os.path.join(DATA_DIR, 'questions-words.txt')


print("DATA_DIR", DATA_DIR)
print("VECTORS_FILE", VECTORS_FILE )
print("WORDS_FILE", WORDS_FILE)

words = []

print("Carregando vetores")
model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)

# for logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Carregando o modelo")
model.accuracy(WORDS_FILE)

i = 1
data = pd.read_csv( os.path.join(DATA_DIR, "palavras.csv"), sep=",", header=0, names=["question_id", "word", "tag"])
for index, row in data.iterrows():    
    if row["tag"] != "NN":
        print("Processando", row["word"], "...")
        syns = model.most_similar(positive= [ row["word"] ])
        i = i + 1 
        syns.extend(row["word"])
        words.append(syns)     
    
    if(i % 1000 == 0):
        print(i, "palavras processadas")
        
df = pd.DataFrame(words)
df.to_csv( os.path.join(DATA_DIR, "words_embeddeds.csv"), index=False)

print("Finalizado")