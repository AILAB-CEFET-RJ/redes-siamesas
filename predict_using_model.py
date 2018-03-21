import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.preprocessing.image import img_to_array, load_img

from utils import calc, dados

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


DATA_DIR = os.environ["DATA_DIR"]
VQA_IMAGES = os.path.join(DATA, "mscoco")
IMAGENET_DIR = os.path.join(DATA, "imagenet")
MODELS_DIR = os.path.join(DATA_DIR, "models")

logger.debug("DATA_DIR : %s", DATA_DIR)
logger.debug("VQA_IMAGES : %s", VQA_IMAGES)
logger.debug("IMAGENET_DIR : %s", IMAGENET_DIR)
logger.debug("MODELS_DIR : %s", MODELS_DIR)


#################################################################
#                       Inicio da execucao                      #
#################################################################

logger.info("Carregando imagens")
logger.debug("Imagem original : %s ", os.path.join(VQA_IMAGES, "train2014/COCO_train2014_000000025162.jpg"))
original_image = plt.imread(os.path.join(VQA_IMAGES, "train2014/COCO_train2014_000000025162.jpg")).astype(np.float32)
original_image = imresize(original_image, (224,224))
original_image = np.divide(original_image, 256)
original_image =  np.expand_dims(original_image, axis=0)


logger.debug("Imagem candidata : %s ", os.path.join(IMAGENET_IMAGES, "n01322604_10456.jpg"))
candidate_image = plt.imread(os.path.join(IMAGENET_DIR, "n01322604_10456.jpg")).astype(np.float32)
candidate_image = imresize(image, (224, 224))
candidate_image = np.divide(image, 256)
candidate_image =  np.expand_dims(image, axis=0)


logger.info("calculando distancia")
distance = calc.euclidian_distance([original_image, candidate_image])
logger.debug("distancia : %d", distance)