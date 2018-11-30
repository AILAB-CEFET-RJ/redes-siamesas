import json, os, sys
import pandas as pd

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
QUESTIONS_DIR = os.path.join(DATA_DIR, "Questions_Train_mscoco")

print("DATA_DIR", DATA_DIR)
print("ANNOTATION_DIR", ANNOTATION_DIR)
print("QUESTIONS_DIR", QUESTIONS_DIR)

print("Carregando as perguntas abertas")
open_questions = json.load(open(os.path.join(QUESTIONS_DIR, "OpenEnded_mscoco_train2014_questions.json")))
print("pronto")

print("Carregando as perguntas abertas")
choice_questions = json.load(open(os.path.join(QUESTIONS_DIR, "MultipleChoice_mscoco_train2014_questions.json")))
print("pronto")

"""
print("Carregando as perguntas multiplas escolhas")
selected_question = pd.read_csv(os.path.join(DATA_DIR, "binary_question_cut.csv"))
print("pronto")
"""
questions_ids_cut = selected_question["question_id"].values


for question in open_questions["questions"]:
    if(question["question_id"] in questions_ids_cut ):
        print(question)