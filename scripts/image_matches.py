import os, sys
import pandas as pd
import mysql.connector


DATA_DIR = os.environ["DATA_DIR"]
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predicoes")
##################################################################################################################
def get_vqa_image_id(filename):
    segmentos = filename.split("_")
    segmentos = segmentos[2].split(".")
    return segmentos[0]

def get_imagenet_image_id(filename):
    segmentos = filename.split("/")
    segmentos = segmentos[1].split(".")
    return segmentos[0]

def insert_match(vqa_id, imagenet_id):
    try:
        update_url = "INSERT INTO img_match (vqa_img_id , imagenet_img_id) values (%s, %s)"
        data = (vqa_id, imagenet_id)
        cursor.execute(update_url, data)
    except mysql.connector.Error as err:
        print(err)
        print("data", vqa_id, imagenet_id)
        sys.exit() 
    except mysql.connector.errors.DataError as err:
        print(err)
        print("data",vqa_id, imagenet_id)
        sys.exit()
        


##################################################################################################################
cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

i = 0
for predictions in os.listdir(PREDICTIONS_DIR):
    print("Carrendo predicoes", predictions)
    df = pd.read_csv( os.path.join(PREDICTIONS_DIR, predictions), names=["vqa", "imagenet", "similar"], header=0)
       
    for index, row in df.iterrows():
        vqa_id = int(get_vqa_image_id(row["vqa"]))
        imagenet_id = get_imagenet_image_id(row["imagenet"])

        insert_match(vqa_id, imagenet_id)

        i += 1
        if i % 1000 == 0:
            print(i, "linhas processadas")
    
    cnx.commit()
print("pronto", i, "linhas processadas")

cursor.close()
cnx.close()
print("finalizado")