import os

### Usar quando as placas de video estiverem ocupadas com outros processos
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from xgboost import XGBClassifier
import itertools
import pandas as pd
from sklearn.utils import shuffle

import logging

#################################################################
#               Configurando logs de execuçao                   #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/train_siamese.log',
                    filemode='w')
logger = logging.getLogger(__name__)

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

def criar_triplas(image_dir, lista_imagens):        
    data = pd.read_csv(lista_imagens, sep=",", header=1, names=["image_id","filename","category_id"])
    image_cache = {}
    for index, row in data.iterrows():
        id = row["filename"]
        if(id in image_cache):
            image_cache[id]["categories"].append(row["category_id"])
        else:
            image_cache[id] = {"image_id" : id, "filename" : row["filename"], "categories" : [row["category_id"]]}
    #Triplas que serao retornadas
    triplas = []

    for index, row in image_cache.items():
        for _i, _r in image_cache.items():
            if index != _i:
                _match = set(row["categories"]).intersection(_r["categories"])
                if len(_match) > 0:
                    triplas.append((row["filename"], _r["filename"], 1))
                else:
                    triplas.append((row["filename"], _r["filename"], 0))
    return shuffle(triplas)

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
    xdata, ydata = [], []
    
    for image_triple in triplas:
        X1 = vec_dict[image_triple[0]]
        X2 = vec_dict[image_triple[1]]
        xdata.append(np.power(np.subtract(X1, X2), 2))
        ydata.append(image_triple[2])
    X, y = np.array(xdata), np.array(ydata)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
   
    return Xtrain, Xtest, ytrain, ytest

#################################################################
#                       validação cruzada                       #
#################################################################

def validacao_cruzada(X, y, clf, k=10):
    best_score, best_clf = 0.0, None
    kfold = KFold(k)
    for kid, (train, test) in enumerate(kfold.split(X, y)):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf.fit(Xtrain, ytrain)
        ytest_ = clf.predict(Xtest)
        score = accuracy_score(ytest_, ytest)
        logger.info("fold {:d}, score: {:.3f}".format(kid, score))
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

lista_imagens = os.path.join(DATA_DIR, 'train_50.csv')
logger.info("Criando triplas")
image_triples = criar_triplas(IMAGE_DIR, lista_imagens)
logger.info("Pronto !!!")

tamanho = len(image_triples)
TAMANHO_LOTE = 100
quantidade_de_lotes = (tamanho // TAMANHO_LOTE) + 1

logger.debug('Triplas criadas: %s', tamanho)
logger.debug('Tamanho do lote: %s', TAMANHO_LOTE)
logger.debug('Quantidade de lotes: %s', quantidade_de_lotes)

logger.info("Iniciando o Pré-processando dados")

Xtrain, Xtest, ytrain, ytest = [], [], [], []

vec_dict = carregar_vetores(VECTOR_FILE)

for i in range(0, quantidade_de_lotes):
    
    logger.debug("Iterando sobre o lote %s", i)
    
    start = i * TAMANHO_LOTE
    end = start + TAMANHO_LOTE - 1
    
    if(i == quantidade_de_lotes):
        amostra = image_triples[start:]
    else:
        amostra = image_triples[start:end]
   
    logger.debug("inicio %s, fim %s", start, end)

    x1, x2, y1, y2 = preprocessar_dados(vec_dict, amostra)
    
    logger.debug("x1: %s", x1)
    
    Xtrain.extend(x1)
    Xtest.extend(x2)
    ytrain.extend(y1)
    ytest.extend(y2)

logger.info("Pre-processamento completo")

Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)

logger.debug("%s %s %s %s", Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

#################################################################
#                         Classificador                         #
#################################################################

logger.info('Iniciando classficador')

clf = XGBClassifier()
best_clf, best_score = validacao_cruzada(Xtrain, ytrain, clf)
scores[3, 2] = best_score
test_report(best_clf, Xtest, ytest)

logger.debug("Salvando model em %s", DATA_DIR)
save_model(best_clf, get_model_file(DATA_DIR, "resnet50", "xgb"))

logger.info("Finalizado com sucesso !!!")
