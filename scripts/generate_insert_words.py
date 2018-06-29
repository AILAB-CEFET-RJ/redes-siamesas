import json, os, sys
import pandas as pd
import csv

DATA_DIR = os.environ["DATA_DIR"]

queries = []

data = pd.read_csv( os.path.join(DATA_DIR, "palavras.csv"), sep=",", header=0, names=["question_id", "word", "tag"])
for index, row in data.iterrows():    
    if row["tag"] == "NN":
        query = "({}, \"{}\", \"{}\")".format(row["question_id"], row["word"], row["tag"])
        queries.append(query)

#INSERT INTO words (question_id, word, role) values 
df = pd.DataFrame(queries)
df.to_csv( os.path.join(DATA_DIR, "insert_words.sql"), index=False, header=False, line_terminator=";\n",quoting=csv.QUOTE_NONE, escapechar="\\")

print("Finalizado")