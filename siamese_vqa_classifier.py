import os
import sys
import argparse
import logging

parser = argparse.ArgumentParser(description='Siamese Network classifier')
parser.add_argument('-d', help='Name of distances file', default='distance.csv')
parser.add_argument('-t', help='Name of triples file', default='triples.csv')
parser.add_argument('-v', help='Name of vectors file', default='vectors.tsv')
parser.add_argument('-n', help='Model o Network (eg. vgg16, vgg19, resnet50, inceptionv3)', default='resnet50')
parser.add_argument('-c', help='Name of classifier method eg. XGBoost', default='xgboost')

parser.add_argument('--batch', help='Size of batch', type=int, default='32')
parser.add_argument('--vs', help='Size of vectors', type=int, default='2048')
parser.add_argument('--epochs', help='Number of epochs', type=int, default='10')
parser.add_argument('--logpath', help='Path top logs', default='/home/ramon/logs/')

args = parser.parse_args()

LOG_FILE = os.path.join(args.logpath, "siamese_vqa_classifier.log")

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=LOG_FILE,
                    filemode='w')

DATA_DIR="/media/ramon/dados/dataset/vqa/"
DISTANCE_FILE = os.path.join("data", args.d)
VECTOR_FILE = os.path.join("data", args.v)
TRIPLES_FILE = os.path.join("data", args.t)
BATCH_SIZE = args.batch
NUM_EPOCHS = args.epochs
VECTOR_SIZE = args.vs

logging.debug("DATA_DIR %s", DATA_DIR)
logging.debug("BATCH_SIZE %s", BATCH_SIZE)
logging.debug("NUM_EPOCHS %s", NUM_EPOCHS)
logging.debug("VECTOR_SIZE %s", VECTOR_SIZE)
logging.debug("DISTANCE_FILE %s", DISTANCE_FILE)
logging.debug("TRIPLES_FILE %s", TRIPLES_FILE)
logging.debug("VECTOR_FILE %s", VECTOR_FILE)