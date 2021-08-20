import torch
from torch import nn
import os
import numpy as np
import argparse
from models import *
from xml.etree import ElementTree

def make_directory(base_path : str) -> int :
    """
        Checks if a directory exists and if doesn't creates the directory.

        Args:
        base_path : Directory path which will be created if it doesn't exist.

        Returns 0 if directory exists else 1
    """
    if os.path.exists(base_path) :
        return 0

    # Create the directory since the path doesn't exist.
    os.mkdir(base_path)
    if os.path.exists(base_path) :
        return 0

    # Path doesn't exist as well as directory couldn't be created.
    print("Error : Cannot create desired path : ", base_path)
    return 1

def generate_csv(csv_name : str, weight_matrix : torch.tensor, base_path : str, transpose = False) -> str :
    """
        Generates csv for weights or bias matrix.

        Args:
        csv_name : A string name for csv file which will store the weights.
        weight_matrix : A torch tensor holding weights that will be stored in the matrix.
        base_path : Base path where csv will be stored.
    """
    # Check if base path exists else create directory.
    make_directory(base_path)
    file_path = os.path.join(base_path, csv_name)
    matrix = weight_matrix.numpy().ravel()
    np.savetxt(file_path, matrix, fmt='%1.128f')
    if transpose:
        matrix = weight_matrix.numpy().transpose().ravel()
        np.savetxt(file_path, matrix, fmt='%1.128f')
        print("Transposed")
    return file_path

def extract_weights(layer, layer_index, base_path) -> {} :
    """
        Extracts weights, biases and other parameters required to reproduce
        the same output.

        Args:
        layer : An torch.nn object (layer).
        layer_index : A string determining name of csv file that will be appended to
                      name of layer.
                      Eg. if layer = nn.Conv2d and layer_index = 0
                          csv_filename = Conv_layer_index.csv
        base_path : A string depicting base path for storing weight / bias csv.

        Returns dictionary of parameter description and parameters.

        Exceptions:
        Currently this has only been tested for convolutional and batch-norm layer.
    """
    parameter_dictionary = {}
    if isinstance(layer, nn.Conv2d):
        # The layer corresponds to Convolutional layer.
        # For convolution layer we require weights and biases to reproduce the
        # same result.
        parameter_dictionary["name"] = "Convolution2D"
        parameter_dictionary["input-channels"] = layer.in_channels
        parameter_dictionary["output-channels"] = layer.out_channels
        # Assume weight matrix is never empty for nn.Conv2d()
        parameter_dictionary["has_weights"] = 1
        parameter_dictionary["weight_offset"] = 0
        csv_name = "conv_weight_" + layer_index + ".csv"
        parameter_dictionary["weight_csv"] = generate_csv(csv_name, \
            layer.weight.detach(), base_path)
        if layer.bias != None:
            parameter_dictionary["has_bias"] = 1
            parameter_dictionary["bias_offset"] = 0
            bias_csv_name = "conv_bias_" + layer_index + ".csv"
            parameter_dictionary["bias_csv"] = generate_csv(bias_csv_name, \
                layer.bias.detach(), base_path)
        else:
            parameter_dictionary["has_bias"] = 0
            parameter_dictionary["bias_offset"] = layer.out_channels
            parameter_dictionary["bias_csv"] = "None"
        parameter_dictionary["has_running_mean"] = 0
        parameter_dictionary["running_mean_csv"] = "None"
        parameter_dictionary["has_running_var"] = 0
        parameter_dictionary["running_var_csv"] = "None"
    elif isinstance(layer, nn.BatchNorm2d) :
        # The layer corresponds to Batch Normalization layer.
        # For batchnorm layer we require weights, biases and running mean and running variance
        # to reproduce the same result.
        parameter_dictionary["name"] = "BatchNorm2D"
        parameter_dictionary["input-channels"] = layer.num_features
        parameter_dictionary["output-channels"] = layer.num_features
        # Assume weight matrix is never empty for nn.BatchNorm2d()
        parameter_dictionary["has_weights"] = 1
        parameter_dictionary["weight_offset"] = 0
        csv_name = "batchnorm_weight_" + layer_index + ".csv"
        parameter_dictionary["weight_csv"] = generate_csv(csv_name, \
            layer.weight.detach(), base_path)
        if layer.bias != None:
            parameter_dictionary["has_bias"] = 1
            parameter_dictionary["bias_offset"] = 0
            bias_csv_name = "batchnorm_bias_" + layer_index + ".csv"
            parameter_dictionary["bias_csv"] = generate_csv(bias_csv_name, \
                layer.bias.detach(), base_path)
        else:
            parameter_dictionary["has_bias"] = 0
            parameter_dictionary["bias_offset"] = layer.out_channels
            parameter_dictionary["bias_csv"] = "None"
        # Assume BatchNorm layer always running variance and running mean.
        running_mean_csv = "batchnorm_running_mean_" + layer_index + ".csv"
        parameter_dictionary["has_running_mean"] = 1
        parameter_dictionary["running_mean_csv"] = generate_csv(running_mean_csv, \
            layer.running_mean.detach(), base_path)
        parameter_dictionary["has_running_var"] = 1
        running_var_csv = "batchnorm_running_var_" + layer_index + ".csv" 
        parameter_dictionary["running_var_csv"] = generate_csv(running_var_csv, \
            layer.running_var.detach(), base_path)
    elif (isinstance(layer, nn.Linear)) :
        # The layer corresponds to Convolutional layer.
        # For convolution layer we require weights and biases to reproduce the
        # same result.
        parameter_dictionary["name"] = "Linear"
        parameter_dictionary["input-channels"] = layer.in_features
        parameter_dictionary["output-channels"] = layer.out_features
        # Assume weight matrix is never empty for nn.Linear()
        parameter_dictionary["has_weights"] = 1
        parameter_dictionary["weight_offset"] = 0
        csv_name = "linear_weight_" + layer_index + ".csv"
        parameter_dictionary["weight_csv"] = generate_csv(csv_name, \
            layer.weight.detach(), base_path, True)
        if layer.bias != None:
            parameter_dictionary["has_bias"] = 1
            parameter_dictionary["bias_offset"] = 0
            bias_csv_name = "linear_bias_" + layer_index + ".csv"
            parameter_dictionary["bias_csv"] = generate_csv(bias_csv_name, \
                layer.bias.detach(), base_path)
        else:
            parameter_dictionary["has_bias"] = 0
            parameter_dictionary["bias_offset"] = layer.out_features
            parameter_dictionary["bias_csv"] = "None"
        parameter_dictionary["has_running_mean"] = 0
        parameter_dictionary["running_mean_csv"] = "None"
        parameter_dictionary["has_running_var"] = 0
        parameter_dictionary["running_var_csv"] = "None"
    elif (isinstance(layer, nn.ConvTranspose2d)):
        # The layer corresponds to Transpose Convolution layer.
        parameter_dictionary["name"] = "TransposeConv2D"
        parameter_dictionary["input-channels"] = layer.in_channels
        parameter_dictionary["output-channels"] = layer.out_channels
        parameter_dictionary["has_weights"] = 1
        parameter_dictionary["weight_offset"] = 0
        csv_name = "convTranspose_weight" + layer_index + ".csv"
        parameter_dictionary["weight_csv"] = generate_csv(csv_name, \
            layer.weight.detach(), base_path)
        if layer.bias != None:
            parameter_dictionary["has_bias"] = 1
            parameter_dictionary["weight_offset"] = 0
            bias_csv_name = "convTranspose_bias" + layer_index + ".csv"
            parameter_dictionary["bias_csv"] = generate_csv(bias_csv_name, \
                layer.bias.detach(), base_path)
        else:
            parameter_dictionary["has_bias"] = 0
            parameter_dictionary["bias_offset"] = layer.out_channels
            parameter_dictionary["bias_csv"] = "None"
        parameter_dictionary["has_running_mean"] = 0
        parameter_dictionary["running_mean_csv"] = "None"
        parameter_dictionary["has_running_var"] = 0
        parameter_dictionary["running_var_csv"] = "None"
    else :
        # The layer corresponds to un-supported layer or layer doesn't have trainable
        # parameter. Example of such layers are nn.MaxPooling2d() and nn.SoftMax.
        parameter_dictionary["name"] = "unknown_layer"
        parameter_dictionary["input-channels"] = 0
        parameter_dictionary["output-channels"] = 0
        parameter_dictionary["has_weights"] = 0
        parameter_dictionary["weight_offset"] = 0
        parameter_dictionary["weight_csv"] = "None"
        parameter_dictionary["has_bias"] = 0
        parameter_dictionary["bias_offset"] = 0
        parameter_dictionary["bias_csv"] = "None"
        parameter_dictionary["has_running_mean"] = 0
        parameter_dictionary["running_mean_csv"] = "None"
        parameter_dictionary["has_running_var"] = 0
        parameter_dictionary["running_var_csv"] = "None"
    return parameter_dictionary

def create_xml_tree(parameter_dictionary : dict, root_tag = "layer") -> ElementTree.ElementTree() :
    """
        Creates an XML tree from a dictionary wrapped around root tag.

        Args:
        parameter_dictionary : Dictionary which will be converted to xml tree.
        root_tag : Tag around which elements of dictionary will be wrapped.
                    Defaults to "layer".
    
        Returns : ElementTree.ElementTree() object.
    """
    layer = ElementTree.Element(root_tag)
    for parameter_desc in parameter_dictionary :
        parameter_description = ElementTree.Element(parameter_desc)
        parameter_description.text = str(parameter_dictionary[parameter_desc])
        layer.append(parameter_description)
    return layer

def create_xml_file(parameter_dictionary : dict,
                    xml_path : str,
                    root_tag : str,
                    element_tag : str) -> int :
    """
        Appends layer description to xml file and if xml doesn't exist or is empty, 
        creates an xml file with required headers.

        Args:
        parameter_dictionary : Dictionary containing layer description.
        xml_path : Path where xml file will be stored / created.
        root_tag : Tag around which xml file will be wrapped.
        element_tag : Tag around which each element in dictionary will be wrapped.
    """
   
    if not os.path.exists(xml_path) :
        # Create base xml file.
        f = open(xml_path, "w")
        data = "<" + root_tag + ">" + "</" + root_tag + ">"
        f.write(data)
        f.close()
    layer_description = create_xml_tree(parameter_dictionary, element_tag)
    xml_file = ElementTree.parse(xml_path)
    root = xml_file.getroot()
    layer = root.makeelement(element_tag, parameter_dictionary)
    root.append(layer_description)
    xml_file.write(xml_path, encoding = "unicode")
    return 0

def iterate_over_layers(modules, xml_path, base_path, layer_index, debug : bool) -> int :
    """
        Parses model and generates csv and xml file which will be iterated by C++ translator.
    
        Args:
        modules : PyTorch model for which parameter csv and xml will be created.
        xml_path : Directory where xml with model config will be saved.
        base_path : Directory where csv will be stored.

        Returns 0 if weights are created else return 1.
    """
    for block in modules :
        for layer in block :
            layer_index += 1
            parameter_dict = extract_weights(layer, str(layer_index), base_path)
            create_xml_file(parameter_dict, xml_path, "model", "layer")
            if not os.path.exists(parameter_dict["weight_csv"]) and parameter_dict["has_weights"] == 1:
                print("Creating weights failed!")
                return 1, layer_index
            if debug :
                print("Weights created succesfully for ", parameter_dict["name"], " layer index :", layer_index)
    return 0, layer_index

def parse_model(model, xml_path, base_path, debug : bool) -> int :
    """
        Parses model and generates csv and xml file which will be iterated by C++ translator.
    
        Args:
        model : PyTorch model for which parameter csv and xml will be created.
        xml_path : Directory where xml with model config will be saved.
        base_path : Directory where csv will be stored.

        Returns 0 if weights are created else return 1.
    """
    layer_index = 0
    error, layer_index = iterate_over_layers(model.features, xml_path, base_path, layer_index, debug)
    if error :
        print("An error occured!")
        return 1
    print(layer_index)
    error, layer_index = iterate_over_layers(model.classifier, xml_path, base_path, layer_index, debug)
    if error :
        print("An error occured!")
        return 1
    print(layer_index)
    if debug :
        print("Model weights saved! Happy mlpack-translation.")
    return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate mlpack-loadable training params.')
  parser.add_argument('model', type = str, help = 'Name of model.')
  args = parser.parse_args()
  model = None
  if args.model == 'darknet19' :
    model = Darknet19(True)
    model.eval()
  if args.model == 'yolov1_tiny' :
      model = TinyYOLO()
      model = model.eval()
      input_tensor = torch.rand((1, 3, 224 * 2, 224 * 2))
      generate_csv("./input_tensor.csv", input_tensor, "./")
      output_tensor = model(input_tensor)
      generate_csv("./output_tensor.csv", output_tensor.detach(), "./")
  if args.model == 'denoiseNet' :
      model = DenoiseNet(True)
      model.eval()
  parse_model(model, "./cfg/" + args.model + ".xml", "./models/" + args.model + "/mlpack-weights/", True)

