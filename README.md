The mlpack **examples** repository contains simple example usages of mlpack.
You can take the code here and adapt it into your application, or compile it and
see what it does and play with it.

Each of the examples are meant to be as simple as possible, and they are
extensively documented.

All the notebooks in this repository can be easily run on
https://lab.mlpack.org/.

### 0. Contents

- [0. Contents](#0-contents)
- [1. Overview](#1-overview)
- [2. Building the examples and usage](#2-building-the-examples-and-usage)
- [3. List of examples](#3-list-of-examples)
- [4. Datasets](#4-datasets)
- [5. Setup](#5-setup)

###  1. Overview

This repository contains examples of mlpack usage that can be easily adapted to
various applications.  If you're looking to figure out how to get mlpack working
for your machine learning task, this repository would definitely be a good place
to look, along with the [mlpack
documentation](https://www.mlpack.org/docs.html).

mlpack is a C++ library that provides machine learning support, but it also
provides bindings to other languages, including Python and Julia, and it also
provides command-line programs.

Therefore, this repository contains examples not just in C++ but also in other
languages.  C++ mlpack usage examples are contained in the `c++/` directory;
Python examples in the `python/` directory, command-line examples in the
`command-line/` directory, and so forth.

### 2. Building the examples and usage

In order to keep this repository as simple as possible, there is no build
system, and all examples are minimal.  For the C++ examples, there is a Makefile
in each example's directory; if you have mlpack installed on your system,
running `make` should work fine.  Some other examples may also use other
libraries, and the Makefile expects those dependencies to also be available.
See the README in each directory for more information, and see the [main mlpack
repository](https://github.com/mlpack/mlpack) and [mlpack
website](https://www.mlpack.org/) for more information on how to install mlpack.

For Python examples and other-language examples, it's expected that you have
mlpack and its dependencies installed.

Each example should be easily runnable and should perform a simple machine
learning task on a dataset.  You might need to download the dataset first---so
be sure to check any README for the example.

### 3. List of examples

Below is a list of examples available in this repository along with a quick
description (just a little bit more than the title):

 - `lstm_electricity_consumption`: use an LSTM-based recurrent neural network to
   predict electricity consumption

 - `lstm_stock_prediction`: predict Google's historical stock price (daily high
   _and_ low) using an LSTM-based recurrent neural network

 - `spam`: predict whether a mobile phone text message in Indonesian is spam 
    or not using logistic regression

 - `mnist_batch_norm`: use batch normalization in a simple feedforward neural
   network to recognize the MNIST digits

 - `mnist_cnn`: use a convolutional neural network (CNN) similar to LeNet-5 to
   recognize the MNIST digits

 - `mnist_simple`: use a very simple three-layer feedforward neural network with
   dropout to recognize the MNIST digits

 - `mnist_vae_cnn`: use a variational autoencoder with convolutional neural
   networks in the encoder and reparametrization networks to recognize the MNIST
   digits
   
 - `neural_network_regression`: use neural network to do regression on Body fat 
    dataset
    
 - `q_learning`: train a simple deep Q-network agent on CartPole environment
   
### 4. Datasets

All the required dataset needed by the examples can be downloaded using the
provided script in the `tools` directory. You will have to execute
`download_data_set.py` from the `tools/` directory and it will download and
extract all the necessary dataset in order for examples to work perfectly:

```sh
cd tools/
./download_data_set.py
```

### 5. Setup
To setup a jupyter local environment that work with C++ using xeus-cling you shall execute the following command:
```sh
./script/jupyter-conda-setup.sh <environment_name>
```