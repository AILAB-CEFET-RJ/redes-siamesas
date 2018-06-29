import mysql.connector
import os, sys, json

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
IMAGE_DIR = os.path.join(DATA_DIR,"vqa", "mscoco")

def insert_category(image_id, category_id, filename):
    try:
        update_url = "INSERT INTO annotation (img_id , wnid, filename, is_valid) values (%s, %s, %s, %s)"
        data = (image_id, category_id, filename, 1)
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", id, image_id, category_id, filename)
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", id, image_id, category_id, filename)

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
    filename = "{}{}".format("000000000000", str(im["image_id"]))
    filename = "COCO_train2014_{}.jpg".format(filename[-12:])
    category_id = im["category_id"]
    
    print(im)
    sys.exit()
    filepath = os.path.join(IMAGE_DIR, filename)

    img_id = im["image_id"]
    
    if( os.path.isfile(filepath) ):
        if img_id not in image_cache:
            insert_category(img_id, category_id, filename)
            image_cache[img_id] = im
    else:
        print(filepath, "not exists")
    
    #insert_category(str(im["id"]), str(im["image_id"]), str(im["category_id"]), filename, im["bbox"])
    
    if(i % 1000 == 0 and i > 0):
        print(i, "/", tam)
        
cnx.commit()
cursor.close()
cnx.close()

print("finalizado")