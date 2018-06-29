import json, sys, os
import mysql.connector

DATA_DIR = os.environ["DATA_DIR"]


cnx = mysql.connector.connect(user='root', password='secret',
                              host='127.0.0.1', port='3306',
                              database='imagenet')




cursor = cnx.cursor()