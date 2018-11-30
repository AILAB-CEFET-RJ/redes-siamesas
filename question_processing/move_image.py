import json, sys, os

DATA_DIR = os.environ["DATA_DIR"]


SOURCE_DIR_TRAIN = os.path.join(DATA_DIR, "vqa", "train2014")
SOURCE_DIR_VAL = os.path.join(DATA_DIR, "vqa", "val2014")
DEST_DIR = os.path.join(DATA_DIR, "vqa", "mscoco")

def move_to(filename, source):
    source_path = os.path.join(source, filename)
    dest_path = os.path.join(DEST_DIR, filename)
    os.rename(source_path, dest_path)

#[ move_to(f, SOURCE_DIR_TRAIN) for f in os.listdir(SOURCE_DIR_TRAIN) ]

[ move_to(f, SOURCE_DIR_VAL) for f in os.listdir(SOURCE_DIR_VAL) ]