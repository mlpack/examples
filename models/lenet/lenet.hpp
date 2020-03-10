/**
 * @file lenet.hpp
 * @author Eugene Freyman
 * @author Daivik Nema
 * @author Kartik Dutt
 * 
 * Definition of LeNet generally used for object detection.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * @code
 * @article{LeCun1998,
 *  author = {Yann LeCun, Leon Bottou, Yoshua Bengio, Pattrick Haffner},
 *  title = {Gradient Based Learning Applied to Document Recognizition},
 *  journal = {IEEE},
 *  year = {1998},
 *  url = {http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_LENET_HPP
#define MODELS_LENET_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Definition of a LeNet CNN.
 */
class LeNet
{
 public:
  //! Create the LeNet object.
  LeNet();

  /**
   * LeNet constructor intializes input shape and number of classes.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param weights One of 'none', 'imagenet'(pre-training on mnist) or path to weights.
   * @param leNetVer Version of LeNet.
   */
  LeNet(const size_t inputChannel,
        const size_t inputWidth,
        const size_t inputHeight,
        const size_t numClasses = 1000,
        const std::string &weights = "none",
        const int leNetVer = 1);

  /**
   * LeNet constructor intializes input shape and number of classes.
   *  
   * @param inputShape A three-valued tuple indicating input shape.
   *                   First value is number of Channels (Channels-First).
   *                   Second value is input height.
   *                   Third value is input width..
   * @param numClasses Optional number of classes to classify images into,
   *                   only to be specified if includeTop is  true.
   * @param weights One of 'none', 'mnist'(pre-training on MNIST) or path to weights.
   * @param leNetVer Version of LeNet.
   */
  LeNet(const std::tuple<size_t, size_t, size_t> inputShape,
        const size_t numClasses = 1000,
        const std::string &weights = "none",
        const int leNetVer = 1);

  //! Get Layers of the model.
  Sequential<>* GetModel() { return leNet; };

  //! Load weights into the model.
  Sequential<>* LoadModel(const std::string& filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filePath);

 private:
  /**
   * Adds Convolution Block.
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  void ConvolutionBlock(const size_t inSize,
                        const size_t outSize,
                        const size_t kernelWidth,
                        const size_t kernelHeight,
                        const size_t strideWidth = 1,
                        const size_t strideHeight = 1,
                        const size_t padW = 0,
                        const size_t padH = 0)
  {
    leNet->Add<Convolution<>>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight);
    leNet->Add<LeakyReLU<>>();

    // Update inputWidth and input Height.
    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
    return;
   }

  /**
   * Adds Pooling Block.
   *
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   */
  void PoolingBlock(const size_t kernelWidth,
                    const size_t kernelHeight,
                    const size_t strideWidth = 1,
                    const size_t strideHeight = 1)
  {
    leNet->Add<MaxPooling<>>(kernelWidth, kernelHeight,
        strideWidth, strideHeight, true);
    // Update inputWidth and inputHeight.
    inputWidth = PoolOutSize(inputWidth, kernelWidth, strideWidth);
    inputHeight = PoolOutSize(inputHeight, kernelHeight, strideHeight);
    return;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param padding The size of the padding (width or height) on one side.
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t padding)
  {
    return std::floor(size + 2 * padding - k) / s + 1;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @return The convolution output size.
   */
  size_t PoolOutSize(const size_t size,
                     const size_t k,
                     const size_t s)
  {
    return std::floor(size - 1) / s + 1;
  }

  //! Locally stored LeNet Model.
  Sequential<> *leNet;

  //! Locally stored LeNet version.
  int leNetVer;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored type of pre-trained weights.
  std::string weights;
}; // class LeNet

} // namespace ann
} // namespace mlpack

#include "lenet_impl.hpp" // Include implementation.

#endif
