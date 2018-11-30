import nltk, sys, os
import pandas as pd
from nltk.tokenize import word_tokenize

DATA_DIR = os.environ["DATA_DIR"]

data = pd.read_csv( os.path.join(DATA_DIR, 'question.csv'), sep=';', header=0, names=['id', 'questions'])

####################################################################
# Aplicação da API                                                 #
#                                                                  #
# Em word_tokenize(row['questions']), estamos pegando a linha      #
# correspondente no DataFrame, que possui uma string, e obtendo um #
# resultado semelhante ao visto com o método split, retornando     #
# assim uma lista.                                                 #
#                                                                  #
# O método pos_tag aplica a categorização (tag) das palavras       #
####################################################################
i = 1
questions = list()
for index, row in data.iterrows():
    tokens = nltk.pos_tag(word_tokenize(row['questions']))
    
    for t in range(0, len(tokens) - 1):
        line = [row["id"], tokens[t][0], tokens[t][1]]
        questions.append(line)

    if(i % 1000 == 0):
        print(i, "perguntas processadas")
    i = i + 1


questions = pd.DataFrame(questions, columns=["question_id", "word", "tag"])

questions.to_csv(os.path.join(DATA_DIR, "palavras_other.csv"), index=False)
print("Finalizado")