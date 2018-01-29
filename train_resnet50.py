import os

### Usar quando as placas de video estiverem ocupadas com outros processos
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scipy.misc import imresize
from keras.applications import resnet50
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "/home/ramon/datasets/vqa/"
IMAGE_DIR = os.path.join(DATA_DIR,"mscoco")



#####################################################################
def image_batch_generator(image_names, batch_size):
    num_batches = len(image_names) // batch_size
    for i in range(num_batches):
        batch = image_names[i * batch_size : (i + 1) * batch_size]
        yield batch
    batch = image_names[(i+1) * batch_size:]
    yield batch


######################################################################
def vectorize_images(image_dir, image_size, preprocessor, 
                     model, vector_file, batch_size=32):
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
#               Generate Vectors using Resnet 50                #
#################################################################

IMAGE_SIZE = 224
VECTOR_FILE = os.path.join(DATA_DIR, "resnet-vectors.tsv")

resnet_model = resnet50.ResNet50(weights="imagenet", include_top=True)
model = Model(inputs=resnet_model.input,
             outputs=resnet_model.get_layer("flatten_1").output)
preprocessor = resnet50.preprocess_input

vectorize_images(IMAGE_DIR, IMAGE_SIZE, preprocessor, model, VECTOR_FILE)