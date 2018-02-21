import os

### Usar quando as placas de video estiverem ocupadas com outros processos
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from scipy.misc import imresize
from keras.applications import resnet50, xception
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from scipy.spatial.distance import pdist 

from xgboost import XGBClassifier
import itertools
import pandas as pd
from sklearn.utils import shuffle

import logging

from utils import calc, dados

#################################################################
#               Configurando logs de execuçao                   #
#################################################################
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/home/rsilva/logs/train_siamese.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

np.random.seed(7)

DATA_DIR = os.environ["DATA_DIR"]
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")

#################################################################
#               Gerando lotes para treinamento                  #
#################################################################
def image_batch_generator(image_names, batch_size):
    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size : (i + 1) * batch_size]
        yield batch
    batch = image_names[(i+1) * batch_size:]
    yield batch


#################################################################
#                       Vetorizando Imagens                     #
#################################################################
def vectorize_images(image_dir, image_size, preprocessor, 
                     model, vector_file, batch_size=32):
    
    if( os.path.isfile(vector_file) ):
        logger.info( "%s already exists ", vector_file)
        return
    
    image_names = os.listdir(image_dir)
    num_vecs = 0
    fvec = open(vector_file, "w")
    for image_batch in image_batch_generator(image_names, batch_size):
        batched_images = []
        for image_name in image_batch:
            image = plt.imread(os.path.join(image_dir, image_name))
            image = imresize(image, (image_size, image_size))
            batched_images.append(image)
        X = preprocessor(np.array(batched_images, dtype="float32"))
        vectors = model.predict(X)
        for i in range(vectors.shape[0]):
            if num_vecs % 100 == 0:
                logger.info("{:d} vectors generated".format(num_vecs))
            image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
            fvec.write("{:s}\t{:s}\n".format(image_batch[i], image_vector))
            num_vecs += 1
    logger.info("{:d} vectors generated".format(num_vecs))
    fvec.close()


#################################################################
#                         Generate Triples                      #
#################################################################

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return pname

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
    #A triplas positivas são a combinação de imagens com a mesma categoria
    for key in img_groups.keys():
        triples = [(x[0], x[1], 1) 
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # é necessário o mesmo número de exemplos negativos
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples 

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
#                 pré-processamento dos dados                   #
#################################################################

def preprocessar_dados(vec_dict, triplas, train_size=0.7):
    """xdata, ydata = [], []
    
    for image_triple in triplas:
        X1 = vec_dict[image_triple[0]]
        X2 = vec_dict[image_triple[1]]      
        distance = calc.distance(X1, X2)       
        xdata.append(distance)
        ydata.append(image_triple[2])
    X, y = np.array(xdata), np.array(ydata)
    """
    ydata = []
    for image_triple in triplas:
        data.append(image_triple[2])
    X = pdist(amostra)
    y = np.array(ydata)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
   
    return Xtrain, Xtest, ytrain, ytest

#################################################################
#                       validação cruzada                       #
#################################################################

def validacao_cruzada(X, y, clf, k=10, best_score=0.0, best_clf=None):    
    kfold = KFold(k)
    for kid, (train, test) in enumerate(kfold.split(X, y)):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf.fit(Xtrain, ytrain)
        ytest_ = clf.predict(Xtest)
        score = accuracy_score(ytest_, ytest)
        logger.debug("fold {:d}, score: {:.3f}".format(kid, score))
        if score > best_score:
            best_score = score
            best_clf = clf
    return best_clf, best_score

#################################################################
#                    Relatório de treino/test                   #
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
#                          Generate Vectors                     #
#################################################################
IMAGE_SIZE = 224
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")

resnet_model = resnet50.ResNet50(weights="imagenet", include_top=True)

model = Model(inputs=resnet_model.input,
             outputs=resnet_model.get_layer("flatten_1").output)
preprocessor = resnet50.preprocess_input

vectorize_images(IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)

#################################################################
#                       Inicio da Execucao                      #
#################################################################


logger.info("*** Iniciando a execução ***")

NUM_VECTORIZERS = 5
NUM_CLASSIFIERS = 4
scores = np.zeros((NUM_VECTORIZERS, NUM_CLASSIFIERS))


TRIPLES_FILES = os.path.join("data/", "triples_train.csv")
#lista_imagens = os.path.join(DATA_DIR, 'train_2014.csv')
logger.info("Carregando triplas")
image_triples = dados.carregar_triplas()
logger.info("Pronto !!!")

tamanho = len(image_triples)
TAMANHO_LOTE = 196
quantidade_de_lotes = (tamanho // TAMANHO_LOTE) + 1

logger.debug('Triplas criadas: %s', tamanho)
logger.debug('Tamanho do lote: %s', TAMANHO_LOTE)
logger.debug('Quantidade de lotes: %s', quantidade_de_lotes)

logger.info("Iniciando o Pré-processando dados")

Xtrain, Xtest, ytrain, ytest = [], [], [], []
X, Y = [], []
vec_dict = carregar_vetores(VECTOR_FILE)

clf = XGBClassifier()
best_clf, best_score = None, 0.0

for i in range(0, quantidade_de_lotes):
    
    logger.info("Iterando sobre o lote %s/%s", i, quantidade_de_lotes)
    
    start = i * TAMANHO_LOTE
    end = start + TAMANHO_LOTE - 1
    
    if(i == quantidade_de_lotes):
        amostra = image_triples[start:]
    else:
        amostra = image_triples[start:end]
                   
    Xtrain, Xtest, ytrain, ytest = preprocessar_dados(vec_dict, amostra)

    logger.info("# Validação cruzada #")
    
    clf.fit(Xtrain, ytrain)
    ytest_ = clf.predict(Xtest)
    score = accuracy_score(ytest_, ytest)
    logger.debug("fold {:d}, score: {:.3f}".format(kid, score))

    if score > best_score:
       best_score = score
       best_clf = clf

    #best_clf, best_score = validacao_cruzada(Xtrain, ytrain, clf, 10, best_score, best_clf)
    scores[3, 2] = best_score
    X.extend(Xtest)
    Y.extend(ytest)
    
test_report(best_clf, Xtest, ytest)
logger.debug("Salvando model em %s", DATA_DIR)
save_model(best_clf, get_model_file(DATA_DIR, "resnet50", "xgb"))

logger.info("Finalizado com sucesso !!!")
