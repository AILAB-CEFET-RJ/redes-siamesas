import json
import mysql.connector
import os
import sys

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
QUESTIONS_DIR = os.path.join(DATA_DIR, "Questions_Train_mscoco")

def save_question(question):   
    insert_into_database(question["question"], question["question_id"], question["image_id"])

def insert_into_database(statement, question_id, image_id):
    try:
        insert_query = "INSERT INTO question (statement, img_id, question_id, answer) values (%s, %s, %s, %s)"
        data = (statement, image_id, question_id, "")
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

cursor = cnx.cursor()

print("Carregando anotacoes")
images = json.load(open(os.path.join(ANNOTATION_DIR, "instances_train2014.json")))
print("pronto")

tam = len(images["annotations"])

print("Carregando as perguntas")
questions = json.load(open(os.path.join(QUESTIONS_DIR, "OpenEnded_mscoco_train2014_questions.json")))
print("pronto")

print("Quantidade de perguntas", len(questions["questions"]))

[save_question(question) for question in questions["questions"]]

cnx.commit()
cursor.close()
cnx.close()

print("finalizado")