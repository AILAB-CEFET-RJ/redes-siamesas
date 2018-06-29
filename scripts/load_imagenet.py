import os, sys
import numpy as np
import pandas as pd
import mysql.connector

DATA_DIR = os.environ["DATA_DIR"]
IMAGENET_DIR = os.path.join(DATA_DIR, "ILSVRC", "Data", "DET", "train", "ILSVRC2013_train")


def insert_category(img):        
    category_id = img[0]
    filename = img[1]
    image_id = "{}_{}".format(category_id, img[2])
    
    try:
        update_url = "INSERT INTO annotation (img_id , wnid, filename, is_valid, dataset_source) values (%s, %s, %s, %s, %s)"
        data = (image_id, category_id, filename, 1, "imagenet")
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", id, image_id, category_id, filename)
        sys.exit()
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", id, image_id, category_id, filename)
        sys.exit()

#################################################################
#               Configurando logs de execucao                   #
#################################################################
def varrer_diretorios(imagenet_dir):
    image_pairs = []
    for synset_dir in os.listdir(imagenet_dir):
        for imagenet_file in os.listdir(os.path.join(IMAGENET_DIR, synset_dir)):
            edges = imagenet_file.split(".")
            edges = edges[0].split("_")
            image_pairs.append( [synset_dir, imagenet_file, edges[1]])
    return image_pairs

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()


lista_imagens = varrer_diretorios(IMAGENET_DIR)

print(len(lista_imagens))

[insert_category(x) for x in lista_imagens]

cnx.commit()
cursor.close()
cnx.close()


print("Salvo")

