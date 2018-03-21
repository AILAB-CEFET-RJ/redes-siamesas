import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################################################################
import time
import threading
import logging
import itertools
import numpy as np
import pandas as pd
from random import shuffle
from utils import calc, dados
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.externals import joblib

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")



#################################################################
#                         Generate Triples                      #
#################################################################
def criar_triplas(lista_imagens):    
    data = pd.read_csv(lista_imagens, sep=",", header=0, names=["image_id","filename","category_id"])
    img_groups = {}
    
    for index, row in data.iterrows():
        pid = row["filename"]
        gid = row["category_id"]
        
        if gid in img_groups:
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    
    pos_triples, neg_triples = [], []
    #A triplas positivas sao a combinacao de imagens com a mesma categoria
    for key in img_groups.keys():
        triples = [(x[0], x[1], 1) 
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # eh necessario o mesmo numero de exemplos negativos
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return pname

#################################################################
#                          Load Vectors                         #
#################################################################

def carregar_vetores(vector_file):
    vec_dict = {}
    fvec = open(vector_file, "r")
    for line in fvec:
        image_name, image_vec = line.strip().split("\t")
        vec = np.array([float(v) for v in image_vec.split(",")])
        vec_dict[image_name] = vec
    fvec.close()
    return vec_dict

#################################################################
#                 pre-processamento dos dados                   #
#################################################################

def preprocessar_dados(vec_dict, triplas, train_size=0.7):
    xdata, ydata = [], []
    i = 0
    tam = len(triplas)
    for image_triple in triplas:
        X1 = vec_dict[image_triple[0]]
        X2 = vec_dict[image_triple[1]]      
        distance = calc.euclidian_distance(X1, X2)       
        xdata.append(distance)
        ydata.append(image_triple[2])
        i = i + 1
        if(i % 1000 == 0):
            logger.info("Processado %d de %d", i,  tam)
    X, y = np.array(xdata), np.array(ydata)
    _Xtrain, _Xtest, _ytrain, _ytest = train_test_split(X, y, train_size=train_size)
    
    Xtrain.extend(_Xtrain)
    Xtest.extend(_Xtest)
    ytrain.extend(_ytrain)
    ytest.extend(_ytest)
    

#################################################################
#                       validacao cruzada                       #
#################################################################

def validacao_cruzada(X, y, clf):    
    clf.fit(X, y)
    ytest_ = clf.predict(Xtest)
    score = accuracy_score(ytest_, ytest)                
    return clf, score

#################################################################
#                    Relatorio de treino/test                   #
#################################################################
def test_report(clf, Xtest, ytest):
    ytest_ = clf.predict(Xtest)
    logger.info("\nAccuracy Score: {:.3f}".format(accuracy_score(ytest_, ytest)))
    logger.info("\nConfusion Matrix")
    logger.info("%s", confusion_matrix(ytest_, ytest))
    logger.info("\nClassification Report")
    logger.info("%s", classification_report(ytest_, ytest))

#################################################################
#                    Salvar / Carregar modelos                  #
#################################################################
def get_model_file(data_dir, vec_name, clf_name):
    return os.path.join(data_dir, "models", "{:s}-{:s}-dot.pkl"
                        .format(vec_name, clf_name))

def save_model(model, model_file):
    joblib.dump(model, model_file)


#################################################################

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/train_siamese.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

NUM_THREADS = 4

NUM_VECTORIZERS = 5
NUM_CLASSIFIERS = 4
scores = np.zeros((NUM_VECTORIZERS, NUM_CLASSIFIERS))

start = time.time()

step_start = time.time()
lista_imagens = os.path.join(DATA_DIR, 'train_2014_1k.csv')
logger.info("gerando triplas")
image_triples = criar_triplas(lista_imagens)
step_elapsed = time.time() - step_start
logger.info("pronto... %s triplas geradas em %s s", len(image_triples), step_elapsed)

logger.info("pre-processando vetores de imagens...")
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")
vec_dict = carregar_vetores(VECTOR_FILE)
logger.info("vetores carregados")

image_triples = np.split(np.array(image_triples), NUM_THREADS)
Xtrain, Xtest, ytrain, ytest = [],[],[],[]

threads = []
for i in range(0,len(image_triples)):
    logger.debug("thread %d iniciada", i)
    t = threading.Thread(target=preprocessar_dados, args=(vec_dict, image_triples))    
    threads.append(t)
    t.start()

logger.debug("Syncronizando as threads")

for thread in threads:
    thread.join()

logger.info("passando pelo classificador")
step_start = time.time()
clf = XGBClassifier()
best_clf, best_score = validacao_cruzada(Xtrain, ytrain, clf)
scores[3, 2] = best_score
step_elapsed = time.time() - step_start

logger.info("pronto... %s em %s", scores, step_elapsed)

test_report(best_clf, Xtest, ytest)
save_model(best_clf, get_model_file(DATA_DIR, "resnet50", "xgb"))
logger.info("modelo salvo")

elapsed = time.time() - start
logger.info("Finalizado com sucesso em %s s", elapsed)
