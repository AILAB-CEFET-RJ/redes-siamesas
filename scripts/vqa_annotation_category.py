import mysql.connector
import os, sys, json

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
IMAGE_DIR = os.path.join(DATA_DIR,"vqa", "mscoco")

def insert_category(image_id, category_id):
    try:
        update_url = "INSERT INTO annotation_category (img_id , category_id) values (%s, %s)"
        data = (image_id, category_id)
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", image_id, category_id)
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", image_id, category_id)

print("Carregando anotacoes...")
data = json.load(open(os.path.join(ANNOTATION_DIR,"instances_train2014.json")))

print("pronto", len(data), "carregadas")

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

tam = len(data["annotations"])
image_cache = {}

for i in range(0, tam-1):
    im = data["annotations"][i]    
    category_id = im["category_id"]    
    img_id = im["image_id"]

    key = "{}.{}".format(img_id, category_id)
    
    if key not in image_cache:
        insert_category(img_id, category_id)    
        image_cache[key] = img_id

    if(i % 1000 == 0 and i > 0):
        print(i, "/", tam)
        
cnx.commit()
cursor.close()
cnx.close()

print("finalizado")