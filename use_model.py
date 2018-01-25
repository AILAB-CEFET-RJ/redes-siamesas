import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import model_from_json
from keras import backend as K
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = "/home/rsilva/datasets"
#DATA_DIR = "/Volumes/Externo/cefet/dataset/"
IMAGE_DIR = os.path.join(DATA_DIR, "vqa")
MODELS_DIR = "models";

gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options))


# load json and create model
json_file = open(os.path.join(MODELS_DIR, "imagenet.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(MODELS_DIR, "imagenet_weights.h5"))
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

original_image = plt.imread(os.path.join(IMAGE_DIR, "train2014/COCO_train2014_000000025162.jpg")).astype(np.float32)
original_image = imresize(original_image, (224,224))
original_image = np.divide(original_image, 256)

image = plt.imread(os.path.join(IMAGE_DIR, "train2014/COCO_train2014_000000025162.jpg")).astype(np.float32)
image = imresize(image, (224, 224))
image = np.divide(image, 256)

pred = loaded_model.predict([original_image, image], batch_size=1, verbose=1)
print(pred)

