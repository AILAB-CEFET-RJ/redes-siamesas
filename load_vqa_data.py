import json
import mysql.connector
import os

DATA_DIR = "/home/ramon/datasets/vqa/"
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations/")
IMAGE_DIR = os.path.join(DATA_DIR,"train2014")

def insert_category(id, image_id, category_id, filename):
    try:
        update_url = "INSERT INTO vqa_images (id, image_id , category_id, filename, year) values (%s, %s, %s, %s, %s)"
        data = (id, image_id, category_id, filename, "2014")
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", id, image_id, category_id, filename)
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", id, image_id, category_id, filename)
        
data = json.load(open(os.path.join(ANNOTATION_DIR,"instances_train2014.json")))

cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

tam = len(data["annotations"])

for i in range(0, tam):
    im = data["annotations"][i]
    filename = "{}{}".format("000000000000", str(im["image_id"]))
    filename = "COCO_train2014_{}.jpg".format(filename[-12:])
    
    filepath = os.path.join(IMAGE_DIR, filename)

    img_id = im["image_id"]

    cache = []

    if( os.path.isfile(filepath) ):
        if img_id not in cache:
            insert_category(str(im["id"]), str(im["image_id"]), str(im["category_id"]), filename)
            cache.append(img_id)
    else:
        print(filepath, "not exists")
    if(i % 100000 == 0):
        print(i, "/", tam)
 
cnx.commit()
cursor.close()
cnx.close()