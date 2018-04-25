import json, sys, os
import mysql.connector

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
QUESTIONS_DIR = os.path.join(DATA_DIR, "Annotations_Train_mscoco")

def save_answer(answer):
    #print(answer)
    #update_question_in_database(answer["answer"], answer["question_id"])
    print(answer["answer_type"], answer["multiple_choice_answer"])
    [print(a) for a in answer["answers"]]
    sys.exit()

def update_question_in_database(answer, question_id):
    try:
        update_query = "UPDATE question SET answer = %s WHERE question_id = %s"
        data = (answer, question_id)
        cursor.execute(update_query, data)
    except mysql.connector.Error as err:
        print(err)
        print(answer, question_id)
    except mysql.connector.errors.DataError as err:
        print(err)
        print(answer, question_id)

print("Carregando as perguntas")
questions = json.load(open(os.path.join(QUESTIONS_DIR, "mscoco_train2014_annotations.json")))
print("pronto")

print("Quantidade de Respostas", len(questions["annotations"]))

cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')

cursor = cnx.cursor()


[save_answer(question) for question in questions["annotations"]]

cnx.commit()
cursor.close()
cnx.close()

print("finalizado")