import os

from keras.layers import Input, Model
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout, Lambda

def base_model(VECTOR_SIZE):
    input_1 = Input(shape=(VECTOR_SIZE,))
    input_2 = Input(shape=(VECTOR_SIZE,))
    merged = Concatenate(axis=-1)([input_1, input_2])

    fc1 = Dense(512, kernel_initializer="glorot_uniform")(merged)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Activation("relu")(fc1)

    fc2 = Dense(128, kernel_initializer="glorot_uniform")(fc1)
    fc2 = Dropout(0.2)(fc2)
    fc2 = Activation("relu")(fc2)

    pred = Dense(2, kernel_initializer="glorot_uniform")(fc2)
    pred = Activation("softmax")(pred)

    model = Model(inputs=[input_1, input_2], outputs=pred)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def carregar_modelo_arquivo(data_dir, vector_name, merge_mode, borf):
    return os.path.join(data_dir, "models", "{:s}-{:s}-{:s}.h5"
                        .format(vector_name, merge_mode, borf))