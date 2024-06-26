The mlpack **examples** repository contains simple example usages of mlpack.
You can take the code here and adapt it into your application, or compile it and
see what it does and play with it. Each of the examples are meant to be as
simple as possible, and they are extensively documented.

All the notebooks in this repository can be easily run on
https://lab.mlpack.org/.

mlpack is a C++ library that provides machine learning support, but it also
provides bindings to other languages, including Python and Julia, and it also
provides command-line programs, see the [main mlpack
repository](https://github.com/mlpack/mlpack) and [mlpack
website](https://www.mlpack.org/) for more information on how to install mlpack.

### 0. Contents

  1. [Overview](#1-overview)
  2. [Building the examples and usage](#2-Building-the-examples-and-usage)
  3. [Datasets](#3-datasets)

###  1. Overview

This repository contains examples of mlpack usage that can be easily adapted to
various applications.  If you're looking to figure out how to get mlpack working
for your machine learning task, this repository would definitely be a good place
to look, along with the [mlpack
documentation](https://www.mlpack.org/docs.html).

Therefore, this repository contains examples that are using common datasets.
This repository is organized per language and per method that is used. 

* `cpp/`: various mlpack C++ examples showing different machine learning
  algorithms. 
* `jupyter_notebook/`: mlpack examples C++ or Python written in jupyter
  notebook format.
* `embedded/`: directory contains mlpack C++ examples with more focus on
  embedded system in the case of compilation and optimized binary and sensor
  input. 
* `cli/` directory contains mlpack methods executed directly from the terminal
  command line, suitable if you have a ready to use dataset and you do not want
  to jump to the code.

### 2. Building the examples and usage

In order to keep this repository as simple as possible, there is no build
system, and all examples are minimal.  For the C++ examples, there is a Makefile
in each example's directory; if you have mlpack installed on your system,
running `make` should work fine.  Rarely, some other examples may also use other
libraries, and the Makefile expects those dependencies to also be available.
See the README in each directory for more information, For Python examples and
other-language examples, it's expected that you have mlpack and its
dependencies installed.

Each example should be easily runnable and should perform a simple machine
learning task on a dataset.  You might need to download the dataset first---so
be sure to check any README for the example.

### 3. Datasets

All the required dataset needed by the examples can be downloaded using the
provided script in the `scripts` directory. You will have to execute
`download_data_set.py` from the `scripts/` directory and it will download and
extract all the necessary datasets inside the `data/` directory in order for
examples to work perfectly:

```sh
cd scripts/
./download_data_set.py
```
