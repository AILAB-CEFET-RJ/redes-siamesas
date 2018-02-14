import json
import mysql.connector
import os

DATA_DIR = "/media/ramon/Dados1/datasets/vqa/"
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations/")
IMAGE_DIR = os.path.join(DATA_DIR,"train2014")

def insert_category(category_id, label, supercategory):
    try:
        update_url = "INSERT INTO vqa_categories (category_id, label, supercategory) values (%s, %s, %s)"
        data = (category_id, label, supercategory)
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", category_id, label, supercategory)
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", category_id, label, supercategory)

data = json.load(open(os.path.join(ANNOTATION_DIR,"instances_train2014.json")))

tam = len(data["categories"])
categories = {}

cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

for i in range(0, tam):
    cat = data["categories"][i] 
    category_id = cat["id"]
    label = cat["name"]
    supercategory = cat["supercategory"]
    insert_category(str(category_id), label, supercategory)

cnx.commit()
cursor.close()
cnx.close()
print("Finalizado")
    