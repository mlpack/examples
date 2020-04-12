#!/bin/bash

echo "This script will download the entire dataset for all the available examples"
echo "Create a dataset directory"
mkdir -p data

pushd data
echo "Start downloading the mnist dataset"
curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --output train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --output train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  --output t10k-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  --output t10k-labels-idx1-ubyte.gz

echo "Extract the downloaded images"
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

echo "Convert mnist images into csv format"
../tools/convert_mnist_2_csv.py train-images-idx3-ubyte.gz
../tools/convert_mnist_2_csv.py train-labels-idx1-ubyte.gz
../tools/convert_mnist_2_csv.py t10k-images-idx3-ubyte.gz
../tools/convert_mnist_2_csv.py t10k-labels-idx1-ubyte.gz

echo "Download the electricty consumption example datasets"
curl https:// --output electricty_consumption_dataset.tar.gz
echo "Extracting"
tar -xzf electricty_consumption_dataset.tar.gz 
echo "Download the stock exchange example datasets"
curl https:// --output stock_exchange.tar.gz
echo "Extracting"
tar -xzf stock_exchange.tar.gz

echo "Cleaning"
rm *.gz

popd
