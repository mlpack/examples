#!/bin/bash
# Download required dataset for each example.

if [ "$1" == "-h" ] || [ "$1" == "--help" ]
then
  echo "Usage: $0 [-n <dataset_name>]"
  echo "Available options:"
  echo "mnist : will download mnist dataset"
  echo "electricty : will download electricty_consumption_dataset"
  echo "stock : will download stock_exchange dataset"
  echo "all : will download all datasets for all examples"
  exit 1
fi

echo "Create a dataset directory"
mkdir -p data
pushd data &>/dev/null

function mnist_dataset() {

  echo "Start downloading the mnist dataset"

  curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output train-images-idx3-ubyte.gz && gunzip train-images-idx3-ubyte.gz

  curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output train-labels-idx1-ubyte.gz && gunzip train-labels-idx1-ubyte.gz

  curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  --output t10k-images-idx3-ubyte.gz && gunzip t10k-images-idx3-ubyte.gz

  curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  --output t10k-labels-idx1-ubyte.gz && gunzip t10k-labels-idx1-ubyte.gz

  echo "Convert mnist images into csv format"
  ../tools/convert_mnist_2_csv.py train-images-idx3-ubyte.gz
  ../tools/convert_mnist_2_csv.py train-labels-idx1-ubyte.gz
  ../tools/convert_mnist_2_csv.py t10k-images-idx3-ubyte.gz
  ../tools/convert_mnist_2_csv.py t10k-labels-idx1-ubyte.gz
}

function electricty_consumption_dataset() {

  echo "Download the electricty consumption example datasets"
  curl https:// --output electricty_consumption_dataset.tar.gz
  
  tar -xzf electricty_consumption_dataset.tar.gz
}

function stock_exchange_dataset() {
  echo "Download the stock exchange example datasets"
  curl https:// --output stock_exchange.tar.gz

  tar -xzf stock_exchange.tar.gz
}

function all_dataset() {
  mnist
  electricty_consumption_dataset
  stock_exchange_dataset
}

function cleanup() {
  rm -rf data
}

popd &>/dev/null

