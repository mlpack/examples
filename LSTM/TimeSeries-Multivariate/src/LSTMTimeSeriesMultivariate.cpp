/**
 * An example of using Recurrent Neural Network (RNN)
 * to make forcasts on a time series of Google stock prices.
 * which we aim to solve using a simple LSTM neural network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file LSTMTimeSeriesMultivariate.cpp
 * @author Zoltan Somogyi based on the work of Mehul Kumar Nirala.
 */

/*
NOTE: the data need to be sorted by date in assending order! The RNN learns the past!

date	close	volume	open	high	low
27-06-16	668.26	2632011	671	672.3	663.284
28-06-16	680.04	2169704	678.97	680.33	673
29-06-16	684.11	1931436	683	687.4292	681.41
30-06-16	692.1	1597298	685.47	692.32	683.65
01-07-16	699.21	1344387	692.2	700.65	692.1301
05-07-16	694.49	1462879	696.06	696.94	688.88
06-07-16	697.77	1411080	689.98	701.68	689.09
07-07-16	695.36	1303661	698.08	698.2	688.215
08-07-16	705.63	1573909	699.5	705.71	696.435
11-07-16	715.09	1107039	708.05	716.51	707.24
...
*/

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;

/*
 * Function to calcute MSE for arma::cube
 */
double MSE1(arma::cube& pred, arma::cube& Y)
{
	double err_sum = 0.0;
	arma::cube diff = pred - Y;
	for (size_t i = 0; i < diff.n_slices; i++)
	{
		mat temp = diff.slice(i);
		err_sum += accu(temp%temp);
	}
	return (err_sum) / (diff.n_elem + 1e-50);
}

/*
 * The time series data for training the model contains the Closing stock price, the Volume of stocks traded,
 * Opening stock price, Highest stock price and Lowest stock price for 'rho' days in the past. 
 * The two target variables (multivariate) we want to predict are the Highest stock price and Lowest stock price
 * (high, low) for the next day! 
 *
 * NOTE: Please note that we do not use the last input data point in the training because there is no target 
 *       (next day (high, low)) for that point!
 */
template<typename InputDataType = arma::mat,
	typename DataType = arma::cube,
	typename LabelType = arma::cube>
	void CreateTimeSeriesData(InputDataType dataset, DataType& X, LabelType& y, size_t rho)
{
	for (size_t i = 0; i < dataset.n_cols - rho; i++)
	{
		X.subcube(span(), span(i), span()) = dataset.submat(span(), span(i, i + rho - 1));
		y.subcube(span(), span(i), span()) = dataset.submat(span(3, 4), span(i + 1, i + rho));
	}
}

/*
 * This function saves the input data for prediction and the prediction results in CSV format. 
 * The prediction results are the (high, low) for the next day and comming from the last slice 
 * from the prediction.
 */
void saveP(const std::string filename, const arma::cube& predictions, data::MinMaxScaler& scale,
	const arma::cube& testX)
{
	std::ofstream out(filename);
	//we need to denormalize (inverse transform) the input data because it was normalized before training
	mat tempX = testX.slice(testX.n_slices - 1);
	scale.InverseTransform(tempX, tempX);

	for (size_t c = 0; c < tempX.n_cols - 1; ++c)
	{
		//the result we are looking for is in the last slice!
		size_t i = predictions.n_slices - 1;
		{
			mat temp = predictions.slice(i);

			// We need to denormalize the predictions also!
			// NOTE: We add 3 extra rows here in order to recreate the input data structure used to 
			// transform the data. This is needed in order to be able to use the right scaling 
			// parameters for the specific column (stock high, low).
			temp.insert_rows(0, 3, 0);
			scale.InverseTransform(temp, temp);
			
			for (size_t r = 0; r < tempX.n_rows; ++r)
			{
				if (r > 0) out << ",";
				out << tempX.at(r, c);
			}

			//let us put one empty column between the input for the prediction and the prediction results
			out << ",,";
			for (size_t r = 3; r < temp.n_rows; ++r)
			{
				if (r > 3) out << ",";
				out << temp.at(r, c);
			}

			//Show the actual result on the screen also.
			//NOTE: Please note that we do not have the last data point in the input for the prediction because 
			//		we did not use it for the training, therefore the prediction result will be for the day before! 
			//		In your own application you may of course load any dataset for prediction!
			if (c == tempX.n_cols - 2)
			{
				cout << std::endl << "The predicted Google stock (high, low) for the last day is the following:" << std::endl;
				for (size_t r = 3; r < temp.n_rows; ++r)
				{
					if (r > 3) cout << ",";
					cout << temp.at(r, c);
				}
				cout << std::endl;
			}
		}
		out << std::endl;
	}

	out.close();
}

int main()
{
	//TODO: change this to your base data path!
	const string basedatapath = "D:/sample-ml-app";

	//TODO: if true the model will be trained; if false the saved model will be read and used for prediction
	//NOTE: training the model may take a long time, therefore once it is trained you can set this to false
	//		and use the model for prediction.
	//NOTE: there is no error checking in this example to see if the trained model exists!
	const bool bTrain = true;

	// Testing data is taken from the dataset in this ratio.
	const double RATIO = 0.1;

	// Number of optimization cycles.
	const int EPOCH = 500;

	// Number of iterations per cycle.
	const int ITERATIONS_PER_EPOCH = 1000;

	// Step size of an optimizer.
	const double STEP_SIZE = 5e-5;

	//number of cells in the LSTM (hidden layers in standard terms)
	//NOTE: you may play with this variable in order to further optimize the model
	const int H1 = 25;

	// Number of data points in each iteration of SGD.
	const size_t BATCH_SIZE = 16;

	// No of timesteps to look in RNN.
	const int rho = 25;

	// Max Rho for LSTM 
	const int maxRho = rho;

	arma::mat dataset;

	// In Armadillo rows represent features, columns represent data points.
	cout << "Reading data ..." << endl;
	data::Load(basedatapath + "/data/Google2016-2019.csv", dataset, true);

	// The CSV file has a header, so it is necessary to remove it. In Armadillo's representation 
	// it is the first column.
	// The first column in the CSV is the date which is not required, therefore removing it 
	// also (first row in in arma::mat).
	dataset = dataset.submat(1, 1, dataset.n_rows - 1, dataset.n_cols - 1);

	// Scale all data into the range (0, 1) for increased numerical stability.
	data::MinMaxScaler scale;
	scale.Fit(dataset);
	scale.Transform(dataset, dataset);

	//we have 5 input data columns and 2 output columns (target)
	size_t inputSize = 5, outputSize = 2;

	//We need to represent the input data for RNN in arma::cube (3D matrix)! The 3rd dimension is the 
	//rho number of past data records the RNN uses for learning.
	arma::cube X, y;
	X.set_size(inputSize, dataset.n_cols - rho + 1, rho);
	y.set_size(outputSize, dataset.n_cols - rho + 1, rho);

	// Create testing and training sets (read the notes above in the function definition!)
	CreateTimeSeriesData(dataset, X, y, rho);

	// Split the data into training and testing sets.
	arma::cube trainX, trainY, testX, testY;
	size_t trainingSize = (1 - RATIO) * X.n_cols;
	trainX = X.subcube(span(), span(0, trainingSize - 1), span());
	trainY = y.subcube(span(), span(0, trainingSize - 1), span());
	testX = X.subcube(span(), span(trainingSize, X.n_cols - 1), span());
	testY = y.subcube(span(), span(trainingSize, X.n_cols - 1), span());

	//only train the model if required	
	if (bTrain) {
		// RNN regression model.
		RNN<MeanSquaredError<>, HeInitialization> model(rho);

		//Model building.
		model.Add<IdentityLayer<> >();
		model.Add<LSTM<> >(inputSize, H1, maxRho);
		model.Add<Dropout<> >(0.5);
		model.Add<LeakyReLU<> >();
		model.Add<LSTM<> >(H1, H1, maxRho);
		model.Add<Dropout<> >(0.5);
		model.Add<LeakyReLU<> >();
		model.Add<LSTM<> >(H1, H1, maxRho);
		model.Add<LeakyReLU<> >();
		model.Add<Linear<> >(H1, outputSize);

		// Setting parameters Stochastic Gradient Descent (SGD) optimizer.
		SGD<AdamUpdate> optimizer(
			STEP_SIZE, // Step size of the optimizer.
			BATCH_SIZE, // Batch size. Number of data points that are used in each iteration.
			ITERATIONS_PER_EPOCH, // Max number of iterations.
			1e-8,// Tolerance.
			true,// Shuffle.
			AdamUpdate(1e-8, 0.9, 0.999)// Adam update policy.
		);

		cout << "Training ..." << endl;

		// Run EPOCH number of cycles for optimizing the solution.
		for (int i = 0; i < EPOCH; i++)
		{
			// Train neural network. If this is the first iteration, weights are
			// random, using current values as starting point otherwise.
			model.Train(trainX, trainY, optimizer);

			// Don't reset optimizer's parameters between cycles.
			optimizer.ResetPolicy() = false;

			arma::cube predOut;
			// Getting predictions on test data points.
			model.Predict(testX, predOut);

			// Calculating mse on test data points.
			double testMSE = MSE1(predOut, testY);
			cout << i + 1 << " - Mean Squared Error := " << testMSE << endl;
		}

		cout << "Finished training." << endl;
		cout << "Saving Model" << endl;
		data::Save(basedatapath + "/saved_models/LSTMMulti.bin", "LSTMMulti", model);
		std::cout << "Model saved in " << basedatapath << "/saved_models/LSTMMulti.bin" << std::endl;
	}

	//NOTE: the below is added in order to show how in a real application the model would be saved, loaded and
	//      then used for prediction.
	//		Please note that we do not have the last data point in testX because we did not use it for the training,
	//		therefore the prediction result will be for the day before! In your own application you may of course 
	//		load any dataset.
	// Load RNN model and use it for prediction
	RNN<MeanSquaredError<>, HeInitialization> modelP(rho);
	std::cout << "Loading model ..." << std::endl;
	data::Load(basedatapath + "/saved_models/LSTMMulti.bin", "LSTMMulti", modelP);
	arma::cube predOutP;
	// Getting predictions on test data points.
	modelP.Predict(testX, predOutP);
	// Calculating mse on prediction.
	double testMSEP = MSE1(predOutP, testY);
	cout << "Mean Squared Error on Prediction data points:= " << testMSEP << endl;

	//save the output predictions
	saveP(basedatapath + "/saved_models/LSTMMulti_predictions.csv", predOutP, scale, testX);

	cout << "Ready!" << std::endl;
	getchar();

	return 0;
}
