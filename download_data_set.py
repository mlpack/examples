#!/usr/bin/python
# Download required dataset for each example.
# Author: Omar Shrit

import argparse
import gzip
import sys
import textwrap
import requests
import shutil

def ungzip(gzip_file, outputfile):
  with open(outputfile, 'wb') as f_in, gzip.open(gzip_file, 'rb') as f_out:
    shutil.copyfileobj(f_out, f_in)  

def mnist_dataset():
  print("Start downloading the mnist dataset")

  train_features = requests.get(
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
  open("train_features.gz", 'wb').write(train_features.content)
  ungzip("train_features.gz", "train_features")
  
  train_labels = requests.get(
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
  open("train_labels.gz", 'wb').write(train_labels.content)
  ungzip("train_labels.gz", "train_labels")

  test_features = requests.get(
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
  open("test_features.gz", 'wb').write(test_features.content)
  ungzip("test_features.gz", "test_features")
  
  test_labels = requests.get(
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
  open("test_labels.gz", 'wb').write(test_labels.content)
  ungzip("test_labels.gz", "test_labels")

  # Still need to Convert mnist images into csv format, not finished yet
 

def electricty_consumption_dataset():
  print("Download the electricty consumption example datasets")
  electricty = request.get("https://www.mlpack.org/datasets/examples/electricity-usage.csv")
  open("data/electricity-usage.csv", 'wb').write(stock.content)

def stock_exchange_dataset():
  print("Download the stock exchange example datasets")
  stock = requests.get("https://www.mlpack.org/datasets/examples/Google2016-2019.csv")
  open("data/stock.csv", 'wb').write(stock.content)

def all_datasets():
  mnist()
  electricty_consumption_dataset()
  stock_exchange_dataset()

# function cleanup() {
#   rm -rf data
# }

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
        electricty : will download electricty_consumption_dataset
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
        mnist_dataset()
      elif args.dataset_name == "electricty":
        electricty_consumption_dataset()
      elif args.dataset_name == "stock":
        stock_exchange_dataset()
      elif args.dataset_name == "all":
        all_datasets()
 
