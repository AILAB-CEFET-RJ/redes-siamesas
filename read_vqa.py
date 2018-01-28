import numpy as np
import pandas as pd
import os

DATA_DIR = "/home/ramon/datasets/vqa/"
IMAGE_DIR = os.path.join(DATA_DIR,"vqa2017")

data = pd.read_csv(os.path.join(DATA_DIR, 'train2014_500.csv'), sep=",", header=1, names=["img_id", "category_id", "filename"])  
print(data.head(5))
"""
image_cache = {}
for index, row in data.iterrows():
    id = row["img_id"]
    if(id in image_cache):
        image_cache[id]["categories"].append(row["category_id"])
    else:
        image_cache[id] = {"img_id" : id, "filename" : row["filename"], "categories" : [row["category_id"]]}

triplas = []

for index, row in image_cache.items():
    for _i, _r in image_cache.items():
        if index != _i:
            _match = set(row["categories"]).intersection(_r["categories"])
            if len(_match) > 0:                
                triplas.append((row["filename"], _r["filename"], 1))
            else:
                triplas.append((row["filename"], _r["filename"], 0))
    

[print(x) for x in triplas[0:5]]"""