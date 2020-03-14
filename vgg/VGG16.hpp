/**
 * @file VGG16.hpp
 * @author Adithya T P (pickle-rick)
 * 
 * Implementation of VGG16 using mlpack.
 * 
 * For more information, see the following paper.
 * 
 * @code
 * @misc{
 *   author = {Karen Simonyan, Andrew Zisserman},
 *   title = {Very Deep Convolutional Networks For Large-Scale Image Recognition},
 *   year = {2015}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_VGG16_HPP
#define MODELS_VGG16_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

using namespace mlpack;
using namespace mlpack::ann;

class VGG16
{
    public:
        // Create object for the Vgg16 network.
        VGG16();

        /**
         * VGG16 constructor intializes input shape, number of classes
         * and weights file.
         *
         * @param inputChannels Number of input channels of the input image.
         * @param inputWidth Width of the input image.
         * @param inputHeight Height of the input image.
         * @param numClasses Optional number of classes to classify images into,
         *                   only to be specified if includeTop is  true.
         * @param includeTop whether to include the fully-connected layer at 
         *                   the top of the network.
         * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
         */
        VGG16(const size_t inputWidth,
              const size_t inputHeight,
              const size_t inputChannel,
              const size_t numClasses,
              const bool includeTop = true,
              const std::string &weights = "None");
        
        /**
         * vgg16 constructor intializes input shape, number of classes
         * and weights file.
         *  
         * @param inputShape A three-valued tuple indicating input shape.
         *                   First value is number of Channels.
         *                   Second value is input height.
         *                   Third value is input width.
         * @param numClasses Optional number of classes to classify images into,
         *                   only to be specified if includeTop is  true.
         * @param includeTop whether to include the fully-connected layer at 
         *                   the top of the network.
         * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
         */
        VGG16(const std::tuple<size_t, size_t, size_t> inputShape,
              const size_t numClasses = 1000,
              const bool includeTop = true,
              const std::string &weights = "None");
        
        // Custom Destructor.
        ~VGG16()
        {
            delete vgg16;
        }

        /** 
         * Defines Model Architecture.
         * 
         * @return Sequential Pointer to the sequential VGG16 model.
         */
        Sequential<> *CompileModel();

        /**
         * Load model from a path.
         * 
         * @param filePath Path to load the model from.
         * @return Sequential Pointer to a sequential model.
         */
        Sequential<> *LoadModel(const std::string &filePath);

        /**
         * Save model to a location.
         *
         * @param filePath Path to save the model to.
         */
        void SaveModel(const std::string &filePath);

         /**
         * Return output shape of model.
         * @returns outputShape of size_t type.
         */
        size_t GetOutputShape() 
        {
            return outputShape;
        };

        /**
         * Returns compiled version of model.
         * If called without compiling would result in empty Sequetial
         * Pointer.
         * 
         * @return Sequential Pointer to a sequential model.
         */
        Sequential<> *GetModel() 
        {
            return vgg16;
        };

        private:
            /**
             * Returns VGG Block.
             * 
             * @param numConv Number of Convolution Blocks in the VGG Block
             * @param numInFilters Number of input maps.
             * @param numOutFilters Number of output maps.
             * @param convKernelWidth Width of the convolutional filter/kernel.
             * @param convKernelHeight Height of the convolutional filter/kernel.
             * @param poolKernelWidth Width of the pooling filter/kernel.
             * @param poolKernelHeight Height of the pooling filter/kernel.
             * @param convStrideWidth Stride of convolutional filter in the x direction.
             * @param convStrideHeight Stride of convolutional filter in the y direction.
             * @param poolStrideWidth Stride of pooling filter in the x direction.
             * @param poolStrideHeight Stride of pooling filter in the y direction.
             * @param padW Padding width of the input.
             * @param padH Padding height of the input.
             */
            void VGGBlock(const size_t numConv,
                          const size_t numInFilters,
                          const size_t numOutFilters,
                          const size_t convKernelWidth,
                          const size_t convKernelHeight,
                          const size_t poolKernelWidth,
                          const size_t poolKernelHeight,
                          const size_t convStrideWidth = 1,
                          const size_t convStrideHeight = 1,
                          const size_t poolStrideWidth = 1,
                          const size_t poolStrideHeight = 1,
                          const size_t padWidth = 0,
                          const size_t padHeight = 0)
            {
                vector<size_t> filters = {numInFilters, numOutFilters};
                filters = getFiltersForBlock(numConv, numOutFilters, filters);
                
                vector<size_t> inputWidths = {inputWidth};
                vector<size_t> inputHeights = {inputHeight};

                inputWidths = VGGOutSize(numConv, inputWidths, inputWidth, convKernelWidth, 
                                         poolKernelWidth, convStrideWidth, poolStrideWidth,
                                         padWidth);
                inputHeights = VGGOutSize(numConv, inputHeights, inputHeight, convKernelHeight, 
                                         poolKernelHeight, convStrideHeight, poolStrideHeight,
                                         padHeight);

                // Add numConv number of "Convolutional and Relu" blocks.
                for(size_t i = 0; i < numConv; i++) {
                    vgg16->Add<Convolution<>>(filters[2 * i], filters[2 * i + 1], 
                        convKernelWidth, convKernelHeight, convStrideWidth, 
                        convStrideHeight, padWidth, padHeight, inputWidths[i], 
                        inputHeights[i]);
                    vgg16->Add<ReLULayer<>>();
                }
                
                // Add Pooling layer.
                vgg16->Add<MaxPooling<>>(poolKernelWidth, poolKernelHeight,
                    poolStrideWidth, poolStrideHeight);

                inputWidth = inputWidths[inputWidths.size() - 1];
                inputHeight = inputHeights[inputHeights.size() - 1];
                
                return;        
            }

            vector<size_t> getFiltersForBlock(size_t numConv, size_t numFilters, vector<size_t> &filters) {
                for(size_t i = 0; i < numConv - 1; i++) {
                    filters.push_back(numFilters);
                    filters.push_back(numFilters);
                }
                return filters;
            }

            // Find output dimensions for a VGG Block.
            vector<size_t> VGGOutSize(const size_t numConv,
                              vector<size_t> inpVec,
                              const size_t inputSize,
                              const size_t convKernelSize,
                              const size_t poolKernelSize,
                              const size_t convStride,
                              const size_t poolStride,
                              const size_t padding = 0) 
            {
                size_t ConvOutSize;
                size_t inputSizeTemp = inputSize;
                // Output dimension after convolutions.
                for(size_t i = 0; i < numConv - 1; i++) {
                    ConvOutSize = std::floor(inputSizeTemp + 2 * padding - convKernelSize) / convStride + 1;
                    inpVec.push_back(ConvOutSize);
                    inputSizeTemp = ConvOutSize;
                }
                // Return VGG block output dimension.
                inpVec.push_back(std::floor(ConvOutSize - poolKernelSize) / poolStride + 1);
                return inpVec;
            }
            
            //! Locally stored vgg16 Model.
            Sequential<>* vgg16;
            
            //! Locally stored width of the image.
            size_t inputWidth;
            
            //! Locally stored height of the image.
            size_t inputHeight;

            //! Locally stored number of channels in the image.
            size_t inputChannel;

            //! Locally stored number of output classes.
            size_t numClasses;

            //! Locally stored bool to determine inclusion of final dense layer.
            bool includeTop;

            //! Locally stored type of pre-trained weights.
            std::string weights;

            //! Locally stored output shape of the vgg16
            size_t outputShape;
};

#include "VGG16_impl.hpp"

#endif