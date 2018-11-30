import mysql.connector
import os, sys, json

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
IMAGE_DIR = os.path.join(DATA_DIR,"vqa", "val2014")

def insert_category(category_id, filename, image_name, subset):
    try:
        update_url = "INSERT INTO vqa_image(category, filename, image_name, subset) values (%s, %s, %s, %s)"
        data = (category_id, filename, image_name, subset)
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", category_id, filename, image_name, subset)
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data", category_id, filename, image_name, subset)

print("Carregando anotacoes...")
data = json.load(open(os.path.join(ANNOTATION_DIR,"instances_val2014.json")))

print("pronto", len(data["annotations"]), "carregadas")

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='ramonsilva03')

cursor = cnx.cursor()

tam = len(data["annotations"])
image_cache = {}

for i in range(0, tam-1):
    im = data["annotations"][i]
    filename = "{}{}".format("000000000000", str(im["image_id"]))
    filename = "COCO_val2014_{}.jpg".format(filename[-12:])
    category_id = im["category_id"]
    image_name = str(im["image_id"])
    
    """print(im)
    sys.exit()"""
    filepath = os.path.join(IMAGE_DIR, filename)

    img_id = im["image_id"]
    
    if( os.path.isfile(filepath) ):
        if img_id not in image_cache:
            insert_category(category_id, filename, image_name, "validation")
            image_cache[img_id] = im
    else:
        print(filepath, "not exists")

    if(i % 1000 == 0 and i > 0):
        print(i, "/", tam)
        
cnx.commit()
cursor.close()
cnx.close()

print("finalizado")