#################################################################
#                         Rede Neural Siamesa                   #
#################################################################

import os
import sys
import itertools
import numpy as np
import pandas as pd
from random import shuffle
import logging


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

import argparse
import logging


parser = argparse.ArgumentParser(description='Classificador para Redes Siamesas')
parser.add_argument('-d', help='Name of distance files')
parser.add_argument('-c', help='Name of classifier method eg. XGBoost', default='xgboost')

args = parser.parse_args()

seed = 7
np.random.seed(seed)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',                    
                    filemode='w')

DATA_DIR="/media/ramon/dados/dataset/vqa/"
DISTANCE_FILE = os.path.join("data", args.d)
BATCH_SIZE = 32
NUM_EPOCHS = 10
VECTOR_SIZE = 2048

logging.debug("DATA_DIR %s", DATA_DIR)
logging.debug("BATCH_SIZE %s", BATCH_SIZE)
logging.debug("NUM_EPOCHS %s", NUM_EPOCHS)
logging.debug("VECTOR_SIZE %s", NUM_EPOCHS)

logging.info("Carregando as triplas de imagens...")


#################################################################
#                    Relatório de treino/test                   #
#################################################################
def test_report(clf, Xtest, ytest):
    ytest_ = clf.predict(Xtest)
    logging.info("\nAccuracy Score: {:.3f}".format(accuracy_score(ytest_, ytest)))
    logging.info("\nConfusion Matrix")
    logging.info("%s", confusion_matrix(ytest_, ytest))
    logging.info("\nClassification Report")
    logging.info("%s", classification_report(ytest_, ytest))

#################################################################
#                    Salvar / Carregar modelos                  #
#################################################################
def get_model_file(data_dir, vec_name, clf_name):
    return os.path.join(data_dir, "models", "{:s}-{:s}-dot.pkl"
                        .format(vec_name, clf_name))

def save_model(model, model_file):
    joblib.dump(model, model_file)

logging.info("Reading distance vectros")
df = pd.read_csv(DISTANCE_FILE, sep=",", header=0, names=["distance", "similar"])

values = df.values

X = values[:,0:1]
y = values[:,1].astype(int)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)

clf = XGBClassifier()

logging.info("ajustando o modelo...")

clf.fit(Xtrain, ytrain)

logging.info("pronto !!!")

logging.info("Testando o modelo...")
ytest_ = clf.predict(Xtest)
logging.info("pronto !!!")
score = accuracy_score(ytest_, ytest)

logging.info("Score %s" % score)

test_report(clf, Xtest, ytest)
logging.debug("Salvando model em %s", DATA_DIR)
save_model(clf, get_model_file(DATA_DIR, "resnet50", "xgb"))

logging.info("Finalizado com sucesso !!!")