/**
 * Loads the tiny imagenet dataset (https://github.com/seshuad/IMagenet.git)
 * into arma matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file dataloader.hpp
 * @author Mehul Kumar Nirala
 */

#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <boost/filesystem.hpp>
#include <mlpack/core/data/image_info.hpp>
#include <mlpack/core/data/load_image_impl.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;
namespace fs = boost::filesystem;

/**
 * Dataloader implementation to load tiny-imagenet dataset
 * (https://github.com/seshuad/IMagenet.git) into armadillo
 * matrix to be fed into VGG19 network.
 */
class Dataloader
{
 public:
 	/**
 	* Dataloader constructor.
 	*
 	* @param numClasses Number of classes in the dataset.
 	*/
	Dataloader(size_t numClasses);

	~Dataloader();

	// To store the name of class labels.
	std::map<std::string, size_t> targetName;

	// To store filenames of training and validation images.
	std::vector<std::string> trainX, valX;

	// To store labels of training and validation images.
	std::vector<size_t> trainY, valY;

	// Number of classes in the dataset.
	size_t numClasses;

	/**
	* Loads the name of training files from the tiny imagenet dataset.
	*
	* @param folderPath Path of the training images.
	* @param shuffle Randomly shuffle the training data.
	*/
	void LoadTrainData(const std::string folderPath, bool shuffle = true);

	/**
	* Loads the name of validation files from the tiny imagenet dataset.
	*
	* @param folderPath Path of the training images.
	*/
	void LoadValData(const std::string folderPath);
	
	/**
	* Loads images into armadillo matrix.
	*
	* @param X Matrix where images are loaded.
	* @param y Labels associated with the images
	* @param train Boolean variable to identify the dataset type (i.e train/val).
	* @param limit Number of datapoints to be stored (0 means all datapoints).
	* @param offset Starting point (in trainX, etc.) from where data is loaded.
	*/
	template<typename dataType, typename labelType>
	void LoadImageData(arma::Mat<dataType>& X,
					   arma::Mat<labelType>& y,
					   bool train,
					   size_t limit = 0,
					   size_t offset = 0);
	
};

#include "dataloader_impl.hpp"

#endif
