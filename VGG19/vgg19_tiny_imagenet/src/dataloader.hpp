/**
 * Loads the tiny imagenet dataset (https://github.com/seshuad/IMagenet.git)
 * into arma matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file dataloader.cpp
 * @author Mehul Kumar Nirala
 */

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <boost/filesystem.hpp>
#include <mlpack/core/data/image_info.hpp>
#include <mlpack/core/data/load_image_impl.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::data;
namespace fs = boost::filesystem;

class Dataloader
{
 public:
	Dataloader();
	~Dataloader();
	std::map<std::string, size_t> targetName;
	std::vector<std::string> trainX, valX;
  	std::vector<size_t> trainY, valY;
  	size_t numClasses;

  	void LoadTrainData(const std::string folderPath, bool shuffle = true);

  	void LoadValData(const std::string folderPath);
	
	template<typename dataType, typename labelType>
	void LoadImageData(arma::Mat<dataType>& X,
					   arma::Mat<labelType>& y,
					   bool train,
					   size_t limit = 0,
					   size_t offset = 0);
	
};

Dataloader::Dataloader()
{
	targetName.clear();
	numClasses = 0;
}

Dataloader::~Dataloader()
{
	trainX.clear();
	trainY.clear();
	valX.clear();
	valY.clear();
	targetName.clear();
}

void Dataloader::LoadTrainData(const std::string folderPath, bool shuffle)
{
	size_t count = 1;

	fs::path p(folderPath);
	fs::directory_iterator end_itr;
	
	// cycle through the directory
	for (fs::directory_iterator itr(p); itr != end_itr; ++itr)
	{
	  if (fs::is_directory(itr->path()))
	  {
	    targetName[itr->path().filename().string()] = count;
	    fs::path folder(itr->path().string());
	    fs::path file ("images");
	    for(auto& entry : boost::make_iterator_range(fs
	    		::directory_iterator(folder / file), {}))
	    {
	      trainX.push_back(entry.path().string());
	      trainY.push_back(count);
	    }
	    count++;
	  }
	}
	if(shuffle)
	{
	  auto seed = unsigned ( std::time(0) );

	  std::srand(seed);
	  std::random_shuffle(trainX.begin(), trainX.end());

	  std::srand(seed);
	  std::random_shuffle(trainY.begin(), trainY.end());
	}
	numClasses = count-1;
}

void Dataloader::LoadValData(const std::string folderPath)
{

	std::ifstream dataFile(folderPath+"/val_annotations.txt");
	while (!dataFile.eof())
	{
	  std::string str;
	  std::getline( dataFile, str);
	  std::stringstream buffer(str);
	  std::string temp;
	  std::vector<std::string> values;
	 
	  while (getline( buffer, temp, '\t'))
	    values.push_back(temp.c_str());

	  if (values.size() == 0)
	    break;
	  valX.push_back(folderPath+"/images/"+values[0]);
	  valY.push_back(targetName[values[1]]);
	}
}

template<typename dataType, typename labelType>
void Dataloader::LoadImageData(arma::Mat<dataType>& X,
						       arma::Mat<labelType>& y,
						       bool train,
						       size_t limit,
						       size_t offset)
{
	arma::Mat<unsigned char> colImg;
  	std::vector<size_t> tempY;
	data::ImageInfo info(64, 64, 3);

	std::vector<std::string> dataset;
  	std::vector<size_t> labels;
  	if (train)
  	{
  		dataset = trainX;
  		labels = trainY;
  	}
  	else
  	{
  		dataset = valX;
  		labels = valY;
  	}
  	size_t numFiles;
  	if (limit == 0)
  		numFiles = dataset.size();
  	else
  		numFiles = std::min(dataset.size(), offset+limit);

	data::Load(dataset[offset], X, info, false, true);
	tempY.push_back(labels[offset]);
	for (size_t i = offset; i < numFiles; i++)
	{
	  try
	  {
	    data::Load(dataset[i], colImg, info, false, true);
	    X = arma::join_rows(X, colImg);
	    tempY.push_back(labels[i]);
	  }
	  catch (std::exception& e)
	  {
	    // Ignore.
	  }
   	cout << "Done  with (" << i << "/" << numFiles << ") of data.\n";
	}
	cout<<X.n_rows<<" "<<X.n_cols<<endl;
	y = arma::Mat<size_t>(tempY);
}
