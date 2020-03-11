In mlpack-model-zoo **project**, we provide state of the art **pre-built 
models** and solutions to some standard datasets.

### 0. Contents

  1. [Introduction](#1-introduction)
  2. [Dependencies](#2-dependencies)
  3. [Building-From-Source](#3-building-from-source)
  4. [Running Models](#4-running-models)
  5. [Current Models](#5-current-models)
  6. [Datasets](#6-datasets)

###  1. Introduction

   This repository contains a number of different models implemented in C++ using
   mlpack. To understand more about mlpack refer to the following links. The sample folder 
   contains snippets and smaller models that demonstrate features of mlpack. 
  - [mlpack homepage](https://www.mlpack.org/)
  - [mlpack documentation](https://www.mlpack.org/docs.html)
  - [Tutorials](https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html)

### 2. Dependencies

To run this project you need:

      mlpack
      Armadillo     >= 8.400.0
      Boost (program_options, math_c99, unit_test_framework, serialization,
             spirit)
      CMake         >= 3.3.2
      ensmallen     >= 2.10.0
To install mlpack refer to the [install guide.](https://www.mlpack.org/docs.html)

All of those should be available in your distribution's package manager. If
not, you will have to compile each of them by hand. See the documentation for
each of those packages for more information.

### 3. Building from source

To install this project run the following command.
  
  `mkdir build && cd build && cmake ../`

Use the optional command `-D DEBUG=ON ` to enable debugging.

Run the make file. Use -jN with the following command where N is the number of cores to be used for the build. 
For instance,
  
  `make -j4`

### 4. Running Models

After building the projects all datasets will be unzipped and executables of the model will be made available in 
bin. You can either execute the same file using:

  `./bin/fileName`
  
### 5. Current Models

Currently model-zoo project has the following models implemented:

  - Simple Convolutional Neural Network on MNIST dataset.
  - Multivariate Time Series prediction using LSTM on Google Stock Prices.
  - Univariate Time Series prediction using LSTM on Electricity Consumption Dataset.
  - Variational Auto-Encoder on MNIST dataset.
  - Variational Convolutional Auto-Encoder on MNIST.
  
### 6. Datasets

Model-Zoo project has the following datasets available:

#### 1. MNIST

[MNIST](http://yann.lecun.com/exdb/mnist/)("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. 
Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification 
algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-
value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-
value is an integer between 0 and 255, inclusive.
The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the 
user. The rest of the columns contain the pixel-values of the associated image. For more information refer to this [MNIST Database](http://yann.lecun.com/exdb/mnist/).

#### 2. Google Stock-Prices Dataset

Google Stock-Prices Dataset consists of stock prices for each day from 27th June, 2016 to 27th June, 2019. Each tuple is 
seperated from its adjacent tuple by 1 day. It consists of following rows that indicate opening, closing, volume and high and 
low of stocks associated with Google on that day.

#### 3. Electricity Consumption Dataset

Contains electricity consumption of a city for 2011 to 2012, where each tuple is seperated from its adjacent tuple by 1 day.  
Each tuple has consumption in kWH and binary values for each Off-peak, Mid-peak, On-peak rows.
