import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

from keras import backend as K
from keras.applications import vgg16
from keras.layers import Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint

from sklearn.model_selection import train_test_split
from random import shuffle
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from sklearn.utils import shuffle

DATA_DIR = "/home/rsilva/datasets/vqa/"
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")
LOG_DIR = "/home/rsilva/logs/"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.join(LOG_DIR, 'basic_siamese.log'),
                    filemode='w')
logger = logging.getLogger(__name__)

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return pname

def criar_triplas(image_dir):
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
    # e necessario o mesmo numero de exemplos negativos
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples 
def carregar_imagem(image_name):
    logging.debug("carragendo imagem : %s" % image_name)
    if image_name not in image_cache:
        logging.debug("cache miss")
        image = plt.imread(os.path.join(IMAGE_DIR, image_name)).astype(np.float32)
        image = imresize(image, (224, 224))
        image = np.divide(image, 256)
        image_cache[image_name] = image
    else:
        logging.debug("cache hit")
    return image_cache[image_name]

def gerar_triplas_em_lote(image_triples, batch_size, shuffle=False):
    logging.info("Gerando triplas")
    while True:
        
        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(len(image_triples)))
        else:
            indices = np.arange(len(image_triples))
        shuffled_triples = [image_triples[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size
      
        logging.info("%s batches of %s generated" % (num_batches, batch_size))
 
        for bid in range(num_batches):
            # loop once per batch
            images_left, images_right, labels = [], [], []
            batch = shuffled_triples[bid * batch_size : (bid + 1) * batch_size]
            for i in range(batch_size):
                lhs, rhs, label = batch[i]
                images_left.append(carregar_imagem(lhs))
                images_right.append(carregar_imagem(rhs))              
                labels.append(label)
            Xlhs = np.array(images_left)
            Xrhs = np.array(images_right)
            Y = np_utils.to_categorical(np.array(labels), num_classes=2)
            yield ([Xlhs, Xrhs], Y)

def calcular_distancia(vecs, normalizar=False):
    x, y = vecs
    if normalizar:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)

def formato_saida_distancia(shapes):
    return shapes[0]

def computar_precisao(predicoes, rotulos):
    return rotulos[predicoes.ravel() < 0.5].mean()

def criar_instancia_rede_neural(entrada):
    seq = Sequential()
    
    # CONV => RELU => POOL
    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=entrada))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Flatten => RELU
    seq.add(Flatten())
    seq.add(Dense(500))
    
    return seq

####################### Inicio da Execucao #######################

logger.info("####################### Inicio da Execucao #######################")

logging.info("Gerando triplas")
<<<<<<< HEAD
lista_imagens = os.path.join(DATA_DIR, 'train_2014_1k.csv')
=======
lista_imagens = os.path.join(DATA_DIR, 'train_500.csv')
>>>>>>> 930fc53300200b6122b3c60bd5395775443fed1e
triplas = criar_triplas(lista_imagens)

logging.debug("# triplas de imagens: %d" % len(triplas))

TAMANHO_LOTE = 192 

divisor = int(len(triplas) * 0.7)
dados_treino, dados_teste = triplas[0:divisor], triplas[divisor:]

################### Processamento das Imagens ##################

formato_entrada = (224, 224, 3)
rede_neural = criar_instancia_rede_neural(formato_entrada)

imagem_esquerda = Input(shape=formato_entrada)
imagem_direita  = Input(shape=formato_entrada)

vetor_saida_esquerda = rede_neural(imagem_esquerda)
vetor_saida_direita  = rede_neural(imagem_direita)

distancia = Lambda(calcular_distancia, 
                output_shape=formato_saida_distancia)([vetor_saida_esquerda, vetor_saida_direita])

############# Computando os vetorese de similaridade #############

fc1 = Dense(128, kernel_initializer="glorot_uniform")(distancia)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc1)
pred = Activation("softmax")(pred)
model = Model(inputs=[imagem_esquerda, imagem_direita], outputs=pred)
#model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

NUM_EPOCAS = 10 

image_cache = {}
lote_de_treinamento = gerar_triplas_em_lote(dados_treino, TAMANHO_LOTE, shuffle=True)
lote_de_validacao = gerar_triplas_em_lote(dados_teste, TAMANHO_LOTE, shuffle=False)

num_passos_treinamento = len(dados_treino) // NUM_EPOCAS
num_passos_validacao = len(dados_teste) // NUM_EPOCAS

csv_logger = CSVLogger(os.path.join(LOG_DIR, 'training_epochs.log'))
model_checkpoint = ModelCheckpoint("models/best.hdf", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [csv_logger, model_checkpoint]

historico = model.fit_generator(lote_de_treinamento,
                            steps_per_epoch=num_passos_treinamento,
                            epochs=NUM_EPOCAS,
                            validation_data=lote_de_validacao,
                            validation_steps=num_passos_validacao,
                            callbacks=callbacks_list)

logging.info("Salvando o modelo em disco")
# serialize model to JSON
model_json = model.to_json()
with open("models/vqa.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("models/vqa_weights.h5")
logging.info("Modelo salvo")

logging.info("Finalizado")
