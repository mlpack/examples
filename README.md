The mlpack **examples** repository contains simple example usages of mlpack.
You can take the code here and adapt it into your application, or compile it and
see what it does and play with it.

Each of the examples are meant to be as simple as possible, and they are
extensively documented.

All the notebooks in this repository can be easily run on
https://lab.mlpack.org/.

### 0. Contents

  1. [Overview](#1-overview)
  2. [Building the examples and usage](#2-Building-the-examples-and-usage)
  3. [List of examples](#3-List-of-examples)
  4. [Datasets](#4-datasets)

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

Below is a list of examples available in `examples` repository along with a quick
description (just a little bit more than the title):

 - `avocado_price_prediction_with_linear_regression`: predict the future price of 
   avocado's depending on various features using linear regression
   [CPP notebook]() [Python notebook]()

 - `breast_cancer_wisconsin_transformation_with_pca`: classify into two groups of
   breast cancer using PCA
   [CPP notebook]() [Python notebook]() 

 - `california_housing_price_prediction_with_linear_regression`: predict California
   housing prices depending on various features using linear regression
   [CPP notebook]() [Python notebook]()

 - `cifar10_cnn`: classify images from CIFAR-10 dataset using pretrained CNN
   [CPP]()

 - `cifar10_transformation_with_pca`: form various sets of image clusters from 
   CIFAR-10 dataset using PCA
   [CPP notebook]() [Python notebook]()

 - `contact_tracing_clustering_with_dbscan`: identifying infected people using
   simple contact tracing method DBSCAN
   [CPP notebook]()

 - `dominant-colors-with-kmeans`: find most dominant colours in an image using 
   K-means clustering
   [CPP notebook]()

 - `forest_covertype_prediction_with_random_forests`: classification using Random
   forest on covertype dataset
   [CPP notebook]() [Python notebook]() [Julia notebook]() [Go notebook]()

 - `graduate_admission_classification_with_Adaboost`: predict probability of a 
   student's admission in a particular university by determining the most important 
   factors that contribute to a student's chance of admission and selecting the most 
   accurate model
   [CPP notebook]() [Python notebook]()

 - `loan_default_prediction_with_decision_tree`: predict whether a person's loan
   payment will be defaulted based on various features using decision tree
   [CPP notebook]() [Python notebook]()

 - `lstm_electricity_consumption`: use an LSTM-based recurrent neural network to
   predict electricity consumption 
   [CPP]()

 - `lstm_stock_prediction`: predict Google's historical stock price (daily high
   _and_ low) using an LSTM-based recurrent neural network
   [CPP]() [CPP notebook]()

 - `microchip_quality_control_naive_bayes`: classify whether a micro-chip is 
   suitable for production usage based on results of quality tests using
   Naive Bayes classifier
   [CPP notebook]() [Python notebook]()

 - `mnist_batch_norm`: use batch normalization in a simple feedforward neural
   network to recognize the MNIST digits
   [CPP]()

 - `mnist_cnn`: use a convolutional neural network (CNN) similar to LeNet-5 to
   recognize the MNIST digits
   [CPP]()

 - `mnist_simple`: use a very simple three-layer feedforward neural network with
   dropout to recognize the MNIST digits
   [CPP]()

 - `mnist_vae_cnn`: use a variational autoencoder with convolutional neural
   networks in the encoder and reparametrization networks to recognize the MNIST
   digits
   [CPP]()

 - `movie_lens_prediction_with_cf`: movie recommendation using collaborative filtering
   [CPP notebook]() [Python notebook]()

 - `neural_network_regression`: use neural network to do regression on Body fat 
   dataset
   [CPP]() [CPP notebook]()

 - `pima_indians_diabetes_clustering_with_kmeans`: classify people having diabetes using
   K-means clustering
   [CPP notebook]()

 - `portfolio_optimization`: helps user investing/selecting stocks using MultiObject
   Decomposition Evolutionary Algorithm Differential variant
   [CPP notebook]()

 - `rainfall_prediction_with_random_forest`: predict next day's rainfall using various
   determinants with random forest
   [CPP notebook]() [Python notebook]()

 - `reinforcement_learning_gym`: 
  
    - `acrobot_dqn`: use 3-Step Double DQN with Prioritized Replay to train an agent 
      to get high scores for the Acrobot environment
      [CPP]() [CPP notebook]()

    - `bipedal_walker_sac`: train a soft actor-critic agent to get high scores for
      bipedal environment
      [CPP]() [CPP notebook]()

    - `cartpole_dqn`: train a simple deep Q-network agent on CartPole environment
      [CPP]() [CPP notebook]()

    - `gym`: A copy of openAI gym toolkit with an example of a random agent on 
      CartPole environment
      [CPP]()

    - `lunar_lander_dqn`: train a simple deep Q-network agent to get high scores 
      in lunar lander v2 environment
      [CPP]() [CPP notebook]()

    - `mountain_car_dqn`: train a simple deep Q-network agent to get high scores 
      in mountain car environment
      [CPP]() [CPP notebook]()

    - `pendulum_dqn`: train a simple deep Q-network agent to get high scores 
      in pendulum environment
      [CPP]() [CPP notebook]()

    - `pendulum_sac`: train a simple soft actor-critic agent to get high scores 
      in pendulum environment
      [CPP]() [CPP notebook]()
    

 - `rocket_injector_design`: 
   [CPP notebook]()

 - `salary_prediction_with_linear_regression`: predict salary of employees based
   on years of experience using linear regression
   [CPP notebook]() [Python notebook]()

 - `spam`: predict whether a mobile phone text message in Indonesian is spam 
   or not using logistic regression
   [CLI]()

 - `student_admission_regression_with_logistic_regression`: predict whether a student
   gets addmitted to a particular university based on previous results using logistic
   regression
   [CPP notebook]()
 
   
### 4. Datasets

All the required dataset needed by the examples can be downloaded using the
provided script in the `tools` directory. You will have to execute
`download_data_set.py` from the `tools/` directory and it will download and
extract all the necessary dataset in order for examples to work perfectly:

```sh
cd tools/
./download_data_set.py
```
