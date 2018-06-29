import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.preprocessing.image import img_to_array, load_img
import logging
from xgboost import XGBClassifier
from utils import calc, dados


DATA_DIR = os.environ["DATA_DIR"]

#################################################################
#                       Configurando logger                     #
#################################################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/predict_using_model.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

VQA_DIR = os.path.join(DATA_DIR, "vqa")
VQA_IMAGES = os.path.join(VQA_DIR, "mscoco")
MODELS_DIR = os.path.join(DATA_DIR, "models")

IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet")

logger.debug("DATA_DIR : %s", DATA_DIR)
logger.debug("VQA_IMAGES : %s", VQA_IMAGES)
logger.debug("IMAGENET_DIR : %s", IMAGENET_DIR)
logger.debug("MODELS_DIR : %s", MODELS_DIR)


#################################################################
#                       Inicio da execucao                      #
#################################################################

logger.info("Carregando imagens")
logger.debug("Imagem original : %s ", os.path.join(VQA_IMAGES, "COCO_train2014_000000025162.jpg"))
original_image = plt.imread(os.path.join(VQA_IMAGES, "COCO_train2014_000000025162.jpg")).astype(np.float32)
original_image = imresize(original_image, (224,224))
original_image = np.divide(original_image, 256)
original_image =  np.expand_dims(original_image, axis=0)


logger.debug("Imagem candidata : %s ", os.path.join(IMAGENET_DIR, "images/n01322604_10456.jpg"))
candidate_image = plt.imread(os.path.join(IMAGENET_DIR, "images/n01322604_10456.jpg")).astype(np.float32)
candidate_image = imresize(candidate_image, (224, 224))
candidate_image = np.divide(candidate_image, 256)
candidate_image =  np.expand_dims(candidate_image, axis=0)


logger.info("calculando distancia")
distance = calc.euclidian_distance([original_image, candidate_image])

logger.info("Carregando o classificador")

clf = XGBClassifier()
booster = Booster()
booster.load_model('./model.xgb')
clf._Booster = booster

prediction = clf.predict(distance)

logger.debug("Prediction %s", prediction)

logger.info("Finalizado com sucesso !!!")