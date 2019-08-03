/**
 * Loads the tiny imagenet dataset (https://github.com/seshuad/IMagenet.git)
 * into arma matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 *
 * @file load.cpp
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

void LoadTrainData(const std::string& folderPath,
                   std::vector<std::string>& dataset,
                   std::vector<size_t>& labels,
                   std::map<std::string, size_t>& targetName)
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
      for(auto& entry : boost::make_iterator_range(fs::directory_iterator(folder / file), {}))
      {
        dataset.push_back(entry.path().string());
        labels.push_back(count);
      }
      count++;
    }
  }
}

void LoadValData(const std::string& folderPath,
                 std::vector<std::string>& dataset,
                 std::vector<size_t>& labels,
                 std::map<std::string, size_t>& targetName)
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

    dataset.push_back(folderPath+"/images/"+values[0]);
    labels.push_back(targetName[values[1]]);
  }
}

template<typename dataType, typename labelType>
void LoadData(arma::Mat<dataType>& X,
              arma::Mat<labelType>& y,
              arma::Mat<dataType>& X_val,
              arma::Mat<labelType>& y_val)
{
  std::map<std::string, size_t> targetName;
  std::vector<std::string> trainFiles, valFiles;
  std::vector<size_t> trainY, valY;
  LoadTrainData("./IMagenet/tiny-imagenet-200/train", trainFiles, trainY, targetName);
  LoadValData("./IMagenet/tiny-imagenet-200/val", valFiles, valY, targetName);

  data::ImageInfo info;
  data::Load(trainFiles, X, info, false, true);
  cout<<X.n_rows<<" "<<X.n_cols<<endl;
  data::Load(valFiles, X_val, info, false, true);
  cout<<X_val.n_rows<<" "<<X_val.n_cols<<endl;

  y = arma::Mat<size_t>(trainY);
  y_val = arma::Mat<size_t>(valY);
}

/* //Loading the data.
int main(int argc, char const *argv[])
{
  arma::Mat<unsigned char> X, X_val;
  arma::Mat<size_t> y, y_val;
  LoadData(X, y, X_val, y_val);
  return 0;
}
 */
