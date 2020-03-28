Time Series Forecasting

In this tutorial we will be explaining how to predict time series data. Time series implies data is dependent on time and the dataset is sampled at regular intervals. It can be broady classified into two categories :

  1. Univariate : Only one variable is dependent on time.
  2. Multivariate : There is dependence of multiple variables on time as well as on each other.

We will be creating models for both type of datasets.

To simply the process we will break down the process in the following steps :

  1. Load and preprocess the data.
  3. Create the Model.
  4. Train the model.
  5. Save the model and finally, predict the future.

Before we start, We will be dealing with two datasets here i.e. Google-Stock-Prices dataset and Electricity Dataset.

1. Google-Stock Prices Dataset :

A sample from dataset is shown below. It can be noticed that each column is dependent on time as well as there exists some interdependence between them, hence the dataset is a multivariate dataset.

|date    |close |volume |open  |high    |low    |
|--------|------|-------|------|--------|-------|
|27-06-16|668.26|2632011|671   |672.3   |663.284|
|28-06-16|680.04|2169704|678.97|680.33  |673    |
|29-06-16|684.11|1931436|683   |687.4292|681.41 |

2. Electricity-Consumption Dataset :

A sample from the dataset is shown below. Here only `Consumption` is dependent on time hence this is a univariate dataset.

|DateTime|Consumption kWh|Off-peak|Mid-peak|On-peak |
|--------|---------------|--------|--------|--------|
|11/25/2011 01:00:00|0.39           |1       |0       |0       |
|11/25/2011 02:00:00|0.33           |1       |0       |0       |
|11/25/2011 03:00:00|0.27           |1       |0       |0       |

Getting Started :

1. Loading the dataset

mlpack has an internal dataloader to load files including csv, hdf5, binary and xml files.

```
arma::mat dataset;
data::Load(dataFile, dataset, true);
```

Preprocessing :

a) Since both datasets have headers and the first column refers to date which is not very useful for the model we drop them from the dataset.

```
dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);
```

b) We set input and output size for model.

For Univariate the input and output size will always be 1. However for multivariate we will taking input all columns and predicting high and low.
Hence for multivariate dataset we will use : `size_t inputSize = 5, outputSize = 2;`

c) Splitting the data into training and validation sets.

We avoid using internal Split function to avoid shuffling.
```
arma::mat trainData = dataset.submat(arma::span(),arma::span(0, (1 - RATIO) *
      dataset.n_cols));
arma::mat testData = dataset.submat(arma::span(), arma::span((1 - RATIO) * dataset.n_cols,
      dataset.n_cols - 1));
```

d) Scale the dataset.

mlpack has an excellent support for multiple scalers, here we will be using MinMaxScaler.
NOTE: We only fit on training data.
```
data::MinMaxScaler scale;
scale.Fit(trainData);
scale.Transform(trainData, trainData);
scale.Transform(testData, testData);
```

e) Prepare data for Model.

For LSTM, we will create a function called Time-Series-Data. A generalized form is given below:
Here `predicitionFeatureStart` and `predictionFeatureEnd` determine the columns that will be predicted by the model.
We will use (3,4) for Google-Stock-Prices dataset and (0, 0) for univariate dataset.
`rho` determines the maximum lookback for the model.

```
template<typename InputDataType = arma::mat,
  typename DataType = arma::cube,
  typename LabelType = arma::cube>
  void CreateTimeSeriesData(InputDataType dataset, DataType& X, LabelType& y, size_t rho
                            size_t predicitionFeatureStart, size_t predicitionFeatureEnd)
{
  for (size_t i = 0; i < dataset.n_cols - rho; i++)
  {
    X.subcube(span(), span(i), span()) = dataset.submat(span(), span(i, i + rho - 1));
    y.subcube(span(), span(i), span()) = dataset.submat(span(predicitionFeatureStart,
        predicitionFeatureEnd), span(i + 1, i + rho));
  }
}
```

2. Creating the model.

Generally RNN models work best for time-series data. We can declare an RNN model as shown below:

`RNN<MeanSquaredError<>, HeInitialization> model(rho);`

We then stack layers such as LSTM to increase learnable parameters to make model complete. We can add multiple such layers.

```
model.Add<LSTM<> >(inputSize, H1, maxRho);
model.Add<Dropout<> >(0.5);
model.Add<LeakyReLU<> >();
model.Add<LSTM<> >(H1, H1, maxRho);
.
.
.
model.Add<LeakyReLU<> >();
model.Add<Linear<> >(H1, outputSize);
```

3. Training the model.

To train the model we will be using ensmallen to help us in optimizing the parameters for our model.
We can declare an optimizer using :

```
SGD<AdamUpdate> optimizer(
        STEP_SIZE, // Step size of the optimizer.
        BATCH_SIZE, // Batch size. Number of data points that are used in each iteration.
        trainData.n_cols * EPOCHS, // Max number of iterations.
        1e-8,// Tolerance.
        true, // Shuffle.
        AdamUpdate(1e-8, 0.9, 0.999)); // Adam update policy.
```

We can simply train the model as shown below:
Ensmallen callbacks are useful for optimization, printing progress as well as stoping at minimum loss as shown below.

```
model.Train(trainX,
                trainY,
                optimizer,
                // PrintLoss Callback prints loss for each epoch.
                ens::PrintLoss(),
                // Progressbar Callback prints progress bar for each epoch.
                ens::ProgressBar(),
                // Stops the optimization process if the loss stops decreasing
                // or no improvement has been made. This will terminate the
                // optimization once we obtain a minima on training set.
                ens::EarlyStopAtMinLoss());
```

4. Predicting the Future.

Load the testing data with the help of steps shown above. Then simply use `model.Predict(testData, modelOutput);`
to store output in modelOutput.

For more details take a look at our code and tutorial.txt.