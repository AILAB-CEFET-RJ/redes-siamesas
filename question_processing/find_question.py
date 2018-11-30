import json, os, sys

DATA_DIR = os.environ["DATA_DIR"]
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations_trainval2014", "annotations")
QUESTIONS_DIR = os.path.join(DATA_DIR, "Questions_Train_mscoco")


print("Carregando anotacoes")
images = json.load(open(os.path.join(ANNOTATION_DIR, "instances_train2014.json")))
print("pronto")

tam = len(images["annotations"])

print("Carregando as perguntas")
questions = json.load(open(os.path.join(QUESTIONS_DIR, "OpenEnded_mscoco_train2014_questions.json")))
print("pronto")

print("Quantidade de perguntas", len(questions["questions"]))

dados = []

for question in questions["questions"]:
    if(question["image_id"] == 100037):
        dados.append(question)
    

print(len(dados))
print(dados)

QUESTIONS_DIR = os.path.join(DATA_DIR, "Annotations_Train_mscoco")

print("Carregando as perguntas")
answers = json.load(open(os.path.join(QUESTIONS_DIR, "mscoco_train2014_annotations.json")))
print("pronto")

dados = []
for answer in answers["annotations"]:
    if(answer['image_id'] == 100037):
        dados.append(answer)


print(len(dados))
print(dados)

print("finalizado")