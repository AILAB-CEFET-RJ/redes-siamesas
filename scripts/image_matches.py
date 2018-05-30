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
    update_url = "INSERT INTO img_match (vqa_img_id , imagenet_img_id) values (%s, %s)"
    data = (vqa_id, imagenet_id)
    cursor.execute(update_url, data)
    


##################################################################################################################

cnx = mysql.connector.connect(user='root', password='secret',
                              host='locahost', port='3306',
                              database='imagenet')

cursor = cnx.cursor()

i = 0
for predictions in os.listdir(PREDICTIONS_DIR):
    
    if predictions == "processados":
        continue
    
    print("Carregando predicoes", predictions)
    df = pd.read_csv( os.path.join(PREDICTIONS_DIR, predictions), names=["vqa", "imagenet", "similar"], header=0)

    try:
        for index, row in df.iterrows():
            vqa_id = int(get_vqa_image_id(row["vqa"]))
            imagenet_id = get_imagenet_image_id(row["imagenet"])

            insert_match(vqa_id, imagenet_id)

            i += 1
            if i % 10000 == 0:
                print(i, "linhas processadas")
        
        cnx.commit()
    except:
        print("Movendo o arquivo para processados", predictions)
        os.rename(os.path.join(PREDICTIONS_DIR, predictions), os.path.join(PREDICTIONS_DIR, "processados",  predictions))
print("pronto", i, "linhas processadas")

cursor.close()
cnx.close()
print("finalizado")