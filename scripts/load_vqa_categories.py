import json, os, sys
import mysql.connector

DATA_DIR = os.environ["DATA_DIR"]

def insert_into_database(statement, question_id, image_id):
    try:
        insert_query = "UPDATE question SET answer = %s where question_id = %s"
        data = (statement, image_id, question_id)
        cursor.execute(insert_query, data)
    except mysql.connector.Error as err:
        print(err)
        print(statement, image_id, question_id)
    except mysql.connector.errors.DataError as err:
        print(err)
        print(statement, image_id, question_id)

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

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
    