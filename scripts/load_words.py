import json, os, sys
import mysql.connector
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

def insert_into_database(question_id, word, tag):
    try:
        insert_query = "INSERT INTO words (question_id, word, role) values (%s, %s, %s)"
        data = (question_id, word, tag)
        cursor.execute(insert_query, data)
    except mysql.connector.Error as err:
        print(err)
        print(question_id, word, tag)
    except mysql.connector.errors.DataError as err:
        print(err)
        print(question_id, word, tag)

cursor = cnx.cursor()
i = 1
data = pd.read_csv( os.path.join(DATA_DIR, "palavras.csv"), sep=",", header=0, names=["question_id", "word", "tag"])
for index, row in data.iterrows():
    insert_into_database(row["question_id"], row["word"], row["tag"])
    if i % 1000 == 0:
        print(i, "processados")
        cnx.commit()       
        i = i + 1

cnx.commit()
cursor.close()
cnx.close()
print("Finalizado")
