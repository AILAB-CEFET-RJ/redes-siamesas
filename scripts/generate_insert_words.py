import json, os, sys
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]

queries = []

data = pd.read_csv( os.path.join(DATA_DIR, "palavras.csv"), sep=",", header=0, names=["question_id", "word", "tag"])
for index, row in data.iterrows():
    query = "INSERT INTO words (question_id, word, role) values ({}, '{}', '{}')".format(row["question_id"], row["word"], row["tag"])
    queries.append(query)


df = pd.DataFrame(queries, index=False)
df.to_csv( os.path.join(DATA_DIR, "insert_words.sql"))

print("Finalizado")