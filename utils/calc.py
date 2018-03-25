import keras.backend as K

def euclidian_distance(inputs):
    assert len(inputs) == 2
    u,v = inputs
    return K.eval(K.sqrt(K.sum(K.square(u - v), axis=-1, keepdims=True)))

def distance(a, b):
    return K.eval( K.square( a - b ))