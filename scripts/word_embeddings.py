import os, sys
import gensim
import logging
import pprint

DATA_DIR = os.environ["DATA_DIR"]
VECTORS_FILE = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")
WORDS_FILE = os.path.join(DATA_DIR, 'questions-words.txt')

model = gensim.models.KeyedVectors.load_word2vec_format(VECTORS_FILE, binary=True)

# for logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Load evaluation dataset of analogy task 
model.accuracy(WORDS_FILE)
# execute analogy task like king - man + woman = queen
#pprint.pprint(model.most_similar(positive=['woman', 'king'], negative=['man']))
pprint.pprint(model.most_similar(positive=['shadow']))