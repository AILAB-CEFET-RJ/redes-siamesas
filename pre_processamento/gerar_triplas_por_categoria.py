import os, sys
import itertools
import numpy as np
import pandas as pd
from random import shuffle
import mysql.connector


DATA_DIR = os.environ["DATA_DIR"]

CATEGORY_FILE = os.path.join(DATA_DIR, "vqa_train", "categories_mscoco.csv")

NUM_EXAMPLES = 100

category_list = pd.read_csv(CATEGORY_FILE)

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='ramonsilva03')

categories = []
for index, row in category_list.iterrows():
    categories.append(row["category_id"])

triples = []
for index, row in category_list.iterrows():
    category = row["category_id"]
    
    print("processando a categoria", category)

    # Similares
    cursor = cnx.cursor()
    query_str = "select filename from vqa_image where category = %s order by rand() limit %s"
    cursor.execute(query_str, (category, NUM_EXAMPLES))

    images = []

    for filename in cursor:                
        images.append(filename[0])
        
    
    cursor.close()
    
    for l_img in images:
        for r_img in images:            
            triples.append( [l_img, r_img, 1] )
            
    # Nao Similares
    cursor = cnx.cursor()
    query_str = "select filename from vqa_image where category <> %s order by rand() limit %s"
    cursor.execute(query_str, (category, NUM_EXAMPLES))

    other_images = []
    for filename in cursor:
        other_images.append(filename[0])

    for l_img in images:
        for r_img in other_images:            
            triples.append( [l_img, r_img, 0] )            

shuffle(triples)
df = pd.DataFrame(triples, columns=['left', 'right', 'similar'])
df.to_csv(os.path.join(DATA_DIR, "vqa_train", "triples_3.csv"), mode='a')        
cnx.close()
print("finalizado")
