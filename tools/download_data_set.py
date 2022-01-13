#!/usr/bin/python
# Download required dataset for each example.
# Author: Omar Shrit

import argparse
import gzip
import os
import sys
import tarfile
import textwrap
from tqdm import tqdm
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
    total_size = n
    block_size = 1
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    for i in range(n):
        image = [ord(l.read(1))]
        t.update(1)
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
    t.close()

    total_size = len(images)
    block_size = 1
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
        t.update(1)
    f.close()
    o.close()
    l.close()
    t.close()

def create_dataset_dir():
  if os.path.exists("../data"):
    os.chdir("../data")
  else:
    os.mkdir("../data")
    os.chdir("../data")

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

def progress_bar(outputfile, request):
  total_size = int(request.headers.get('content-length', 0))
  block_size = 1024 #1 Kibibyte
  t=tqdm(total=total_size, unit='iB', unit_scale=True)
  with open(outputfile, 'wb') as f:
    for data in request.iter_content(block_size):
      t.update(len(data))
      f.write(data)
  t.close()
  if total_size != 0 and t.n != total_size:
    print("ERROR, something went wrong")

def mnist_dataset():
  print("Start downloading the mnist dataset")
  train_features = requests.get(
      "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
  progress_bar("train_features.gz", train_features)
  ungzip("train_features.gz", "train_features.ubytes")

  train_labels = requests.get(
      "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
  progress_bar("train_labels.gz", train_labels)
  ungzip("train_labels.gz", "train_labels.ubytes")

  test_features = requests.get(
      "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
  progress_bar("test_features.gz", test_features)
  ungzip("test_features.gz", "test_features.ubytes")

  test_labels = requests.get(
      "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
  progress_bar("test_labels.gz", test_labels)
  ungzip("test_labels.gz", "test_labels.ubytes")

  print("Converting mnist ubytes images files into csv...")
  print("This might take a while...")
  print("Converting features images...")
  convert("train_features.ubytes", "train_labels.ubytes", "mnist_train.csv", 60000)
  print("Converting label images...")
  convert("test_features.ubytes", "test_labels.ubytes", "mnist_test.csv", 10000)
  clean()

def electricity_consumption_dataset():
  print("Download the electricty consumption example datasets")
  electricity = requests.get("https://datasets.mlpack.org/examples/electricity-usage.csv")
  progress_bar("electricity-usage.csv", electricity)

def stock_exchange_dataset():
  print("Download the stock exchange example datasets")
  stock = requests.get("https://datasets.mlpack.org/examples/Google2016-2019.csv")
  progress_bar("Google2016-2019.csv", stock)

def body_fat_dataset():
  print("Download the body fat datasets")
  bodyFat = requests.get("https://datasets.mlpack.org/examples/bodyfat.tsv")
  progress_bar("BodyFat.tsv", bodyFat)

def iris_dataset():
  print("Downloading iris datasets...")
  iris = requests.get("https://datasets.mlpack.org/iris.tar.gz")
  progress_bar("iris.tar.gz", iris)
  tar = tarfile.open("iris.tar.gz", "r:gz")
  tar.extractall()
  tar.close()
  clean()

def spam_dataset():
  print("Downloading spam dataset...")
  spam = requests.get("https://www.mlpack.org/datasets/dataset_sms_spam_bhs_indonesia_v1.tar.gz")
  progress_bar("dataset_sms_spam_bhs_indonesia_v1.tar.gz", spam)
  tar = tarfile.open("dataset_sms_spam_bhs_indonesia_v1.tar.gz", "r:gz")
  tar.extractall()
  tar.close()
  clean()

def salary_dataset():
  print("Downloading salary dataset...")
  salary = requests.get("http://datasets.mlpack.org/Salary_Data.csv")
  progress_bar("Salary_Data.csv", salary)

def cifar10_dataset():
  print("Downloading CIFAR10 dataset...")
  cifar = requests.get("http://datasets.mlpack.org/cifar-10.tar.xz")
  progress_bar("cifar-10.tar.xz", cifar)
  tar = tarfile.open("cifar-10.tar.xz", "r:xz")
  tar.extractall()
  tar.close()
  clean()

def all_datasets():
  mnist_dataset()
  electricity_consumption_dataset()
  stock_exchange_dataset()
  iris_dataset()
  salary_dataset()
  body_fat_dataset()
  spam_dataset()
  cifar10_dataset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Download dataset script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        Dataset downloading script!
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
        iris : will downlaod the iris dataset
        bodyFat : will download the bodyFat dataset
        spam : will download the spam dataset
        salary: will download the salary dataset
        cifar10: will download the cifar10 dataset
        all : will download all datasets for all examples
        '''))

    parser.add_argument('--dataset_name', metavar="dataset name", type=str, help="Enter dataset name to download")
    args = parser.parse_args()

    if len(sys.argv) == 1:
      create_dataset_dir()
      all_datasets()

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
      elif args.dataset_name == "bodyFat":
        create_dataset_dir()
        body_fat_dataset()
      elif args.dataset_name == "spam":
        create_dataset_dir()
        spam_dataset()
      elif args.dataset_name == "salary":
        create_dataset_dir()
        salary_dataset()
      elif args.dataset_name == "cifar10":
        create_dataset_dir()
        cifar10_dataset()
      elif args.dataset_name == "all":
        create_dataset_dir()
        all_datasets()
