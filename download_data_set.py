#!/usr/bin/python
# Download required dataset for each example.
# Author: Omar Shrit

import argparse
import gzip
import os
import sys
import tarfile
import textwrap
import requests
import shutil

''' This function is originally written by Joseph Redmon (pjreddie).
    The original source code can be found here:
      https://pjreddie.com/projects/mnist-in-csv/
'''
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def create_dataset_dir():
  if os.path.exists("data"):
    os.chdir("data")
  else:
    os.mkdir("data")
    os.chdir("data")

def clean():
  dataset = "."
  files = os.listdir(dataset)
  for f in files:
    if f.endswith(".gz"):
      os.remove(os.path.join(dataset, f))
    elif f.endswith(".tar.gz"):
      os.remove(os.path.join(dataset, f))
    elif f.endswith(".ubytes"):
      os.remove(os.path.join(dataset, f))      

def ungzip(gzip_file, outputfile):
  with open(outputfile, 'wb') as f_in, gzip.open(gzip_file, 'rb') as f_out:
    shutil.copyfileobj(f_out, f_in) 

def mnist_dataset():
  print("Start downloading the mnist dataset")
  train_features = requests.get(
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
  open("train_features.gz", 'wb').write(train_features.content)
  ungzip("train_features.gz", "train_features.ubytes")
  
  train_labels = requests.get(
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
  open("train_labels.gz", 'wb').write(train_labels.content)
  ungzip("train_labels.gz", "train_labels.ubytes")

  test_features = requests.get(
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
  open("test_features.gz", 'wb').write(test_features.content)
  ungzip("test_features.gz", "test_features.ubytes")
  
  test_labels = requests.get(
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
  open("test_labels.gz", 'wb').write(test_labels.content)
  ungzip("test_labels.gz", "test_labels.ubytes")

  convert("train_features.ubytes", "train_labels.ubytes",
  "mnist_train.csv", 60000)
  convert("test_features.ubytes", "test_labels.ubytes",
  "mnist_test.csv", 10000)
  clean()

def electricity_consumption_dataset():
  print("Download the electricty consumption example datasets")
  electricity = requests.get("https://www.mlpack.org/datasets/examples/electricity-usage.csv")
  open("electricity-usage.csv", 'wb').write(electricity.content)

def stock_exchange_dataset():
  print("Download the stock exchange example datasets")
  stock = requests.get("https://www.mlpack.org/datasets/examples/Google2016-2019.csv")
  open("stock.csv", 'wb').write(stock.content)

def iris_dataset():
  print("Downloading iris datasets...")
  iris = requests.get("https://www.mlpack.org/datasets/iris.tar.gz")
  open("iris.tar.gz", 'wb').write(iris.content)
  tar = tarfile.open("iris.tar.gz", "r:gz")
  tar.extractall()
  tar.close()
  clean()
  
def all_datasets():
  mnist_dataset()
  electricity_consumption_dataset()
  stock_exchange_dataset()
  iris_dataset()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Download dataset script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        Dataset script!
        --------------------------------
        This script is used to download the dataset required to run
        mlpack examples.
        It can be used to download one dataset at the time,
        or all of them at the same time.

        Usage: --dataset_name dataset_name
        Available options:
        mnist : will download mnist dataset
        electricity : will download electricty_consumption_dataset
        stock : will download stock_exchange dataset
        all : will download all datasets for all examples
        '''))
 
    parser.add_argument('--dataset_name', metavar="dataset name", type=str, help="Enter dataset name to download")
    args = parser.parse_args()

    if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

    if args.dataset_name:
      if args.dataset_name == "mnist":
        create_dataset_dir()
        mnist_dataset()
      elif args.dataset_name == "electricity":
        create_dataset_dir()
        electricity_consumption_dataset()
      elif args.dataset_name == "stock":
        create_dataset_dir()
        stock_exchange_dataset()
      elif args.dataset_name == "iris":
        create_dataset_dir()
        iris_dataset()
      elif args.dataset_name == "all":
        create_dataset_dir()
        all_datasets()
 
