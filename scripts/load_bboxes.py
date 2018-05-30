import json, os, sys
import mysql.connector
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]
BBOXES_DIR = os.path.join(DATA_DIR, "bboxes_imagenet")


cnx = mysql.connector.connect(user='root', password='secret',
                              host='locahost', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

def save_data(data):    
    
    for i in range(0, len(data)):
        filename = data[i][0].split(".")[0]
        class_name = data[i][1]
        x1 = data[i][2]
        x2 = data[i][3]
        y1 = data[i][4]
        y2 = data[i][5]

        try:
            insert_query = "INSERT INTO bound_box (img_id, x1, x2, y1, y2, class) values (%s, %s, %s, %s, %s, %s)"
            insert_data = (filename, x1, x2, y1, y2, class_name)
            cursor.execute(insert_query, insert_data)
        except mysql.connector.Error as err:
            print(err)        
        except mysql.connector.errors.DataError as err:
            print(err)        

for imagenet_file in os.listdir(os.path.join(BBOXES_DIR)):
    try:
        data = pd.read_csv( os.path.join(BBOXES_DIR, imagenet_file), sep=",", header=1, names=["filename","category","x1","x2","y1","y2"])
        print("Salvando dados de ", imagenet_file)
        save_data(data.values)
    except Exception as err:
        print(err)

cnx.commit()
cursor.close()
cnx.close()

print("Finalizado")