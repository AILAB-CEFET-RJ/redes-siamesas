import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import numpy as np
from random import shuffle
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.applications import xception


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


def vectorize_images(image_dir, image_size, preprocessor, 
                     model, vector_file, batch_size=32):
    
    """if( os.path.isfile(vector_file) ):
        print(vector_file,"already exists")
        return
    """
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
                print("{:d} vectors generated".format(num_vecs))
            image_vector = ",".join(["{:.5e}".format(v) for v in vectors[i].tolist()])
            fvec.write("{:s}\t{:s}\n".format(image_batch[i], image_vector))
            num_vecs += 1
    print("{:d} vectors generated".format(num_vecs))
    fvec.close()


#################################################################
#                          Constantes                           #
#################################################################
IMAGE_SIZE = 299
DATA_DIR    = os.environ["DATA_DIR"]
IMAGE_DIR   = os.path.join(DATA_DIR, 'mscoco')
MODELS_DIR  = os.path.join(DATA_DIR, "models")
VECTOR_FILE = os.path.join(MODELS_DIR, "xception-vectors.tsv")


#################################################################
#                       Inicio da Execucao                      #
#################################################################

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/xception_training.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


xception_model = xception.Xception(weights="imagenet", include_top=True)

model = Model(input=xception_model.input,
             output=xception_model.get_layer("avg_pool").output)
preprocessor = xception.preprocess_input

vectorize_images(IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)