#ifndef PLOT_HPP
#define PLOT_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int scatter(const std::string& fname,
            const std::string&#include "../utils/plot.hpp"
 xCol,
            const std::string& yCol,
            const std::string& dateCol = "",
            const std::string& maskCol = "",
            const std::string& type = "",
            const std::string& color = "",
            const std::string& xLabel = "",
            const std::string& yLabel = "",
            const std::string& figTitle = "",
            const int figWidth = 26,
            const int figHeight = 7)
{

  // Calls Python function cscatter and generates a scatter plot of Xcol and yCol and saves it,
  // so the plot can later be imported in C++ notebook using xwidget.

  // PyObject contains info Python needs to treat a pointer to an object as an object.
  // It contains object's reference count and pointer to corresponding object type.
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

  // Initialize Python Interpreter.
  Py_Initialize();
  // Import sys module in Interpreter and add current path to python search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");

  // Import the Python module.
  pName = PyUnicode_DecodeFSDefault("plot");
  pModule = PyImport_Import(pName);

  // Get the reference to Python Function to call.
  pFunc = PyObject_GetAttrString(pModule, "cscatter");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(12);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyString_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature to be plotted along X axis.
  PyObject* pXcol = PyString_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXcol);

  // String object representing the name of the feature to be plotted along Y axis.
  PyObject* pYcol = PyString_FromString(yCol.c_str());
  PyTuple_SetItem(pArgs, 2, pYcol);

  // String object representing the name of the feature to be parsed as TimeStamp.
  PyObject* pDateCol = PyString_FromString(dateCol.c_str());
  PyTuple_SetItem(pArgs, 3, pDateCol);

  // String object representing the name of the feature to be used to mask the plot data points.
  PyObject* pMaskCol = PyString_FromString(maskCol.c_str());
  PyTuple_SetItem(pArgs, 4, pMaskCol);

  // String object representing the value for masking.
  PyObject* pType = PyString_FromString(type.c_str());
  PyTuple_SetItem(pArgs, 5, pType);

  // String object representing the feature name to be used as color value in plot.
  PyObject* pColor = PyString_FromString(color.c_str());
  PyTuple_SetItem(pArgs, 6, pColor);

  // String object representing the X axis label.
  PyObject* pXlabel = PyString_FromString(xLabel.c_str());
  PyTuple_SetItem(pArgs, 7, pXlabel);

  // String object representing the Y axis label.
  PyObject* pYlabel = PyString_FromString(yLabel.c_str());
  PyTuple_SetItem(pArgs, 8, pYlabel);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 9, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 10, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 11, pFigHeight);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int barplot(const std::string& fname,
            const std::string& x,
            const std::string& y,
            const std::string& dateCol = "",
            const std::string& figTitle = "",
            const int figWidth = 5,
            const int figHeight = 7)
{

  // Calls Python function cbarplot and generates a barplot plot of x and y and saves it,
  // so the plot can later be imported in C++ notebook using xwidget.

  // PyObject contains info Python needs to treat a pointer to an object as an object.
  // It contains object's reference count and pointer to corresponding object type.
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

  // Initialize Python Interpreter.
  Py_Initialize();
  // Import sys module in Interpreter and add current path to python search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");

  // Import the Python module.
  pName = PyUnicode_DecodeFSDefault("plot");
  pModule = PyImport_Import(pName);

  // Get the reference to Python Function to call.
  pFunc = PyObject_GetAttrString(pModule, "cbarplot");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(7);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyString_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature to be plotted along X axis.
  PyObject* pX = PyString_FromString(x.c_str());
  PyTuple_SetItem(pArgs, 1, pX);

  // String object representing the name of the feature to be plotted along Y axis.
  PyObject* pY = PyString_FromString(y.c_str());
  PyTuple_SetItem(pArgs, 2, pY);

  // String object representing the name of the feature to be parsed as TimeStamp.
  PyObject* pDateCol = PyString_FromString(dateCol.c_str());
  PyTuple_SetItem(pArgs, 3, pDateCol);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 4, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 5, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 6, pFigHeight);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int heatmap(const std::string& fname,
            const std::string& colorMap,
            const std::string& figTitle = "",
            const int annotation = false,
            const int figWidth = 12,
            const int figHeight = 6)
{

  // PyObject contains info Python needs to treat a pointer to an object as an object.
  // It contains object's reference count and pointer to corresponding object type.
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

  // Initialize Python Interpreter.
  Py_Initialize();
  // Import sys module in Interpreter and add current path to python search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");

  // Import the Python module.
  pName = PyUnicode_DecodeFSDefault("plot");
  pModule = PyImport_Import(pName);

  // Get the reference to Python Function to call.
  pFunc = PyObject_GetAttrString(pModule, "cheatmap");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(6);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyString_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of color map to be used for plotting.
  PyObject* pColorMap = PyString_FromString(colorMap.c_str());
  PyTuple_SetItem(pArgs, 1, pColorMap);

  // Boolean object indicating if correlation values must be annotated in figure.
  PyObject* pAnnotation = PyBool_FromLong(annotation);
  PyTuple_SetItem(pArgs, 2, pAnnotation);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 3, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 4, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 5, pFigHeight);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int lmplot(const std::string& fname,
           const std::string& figTitle,
           const int figWidth = 6,
           const int figHeight = 7)
{

  // PyObject contains info Python needs to treat a pointer to an object as an object.
  // It contains object's reference count and pointer to corresponding object type.
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

  // Initialize Python Interpreter.
  Py_Initialize();
  // Import sys module in Interpreter and add current path to python search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");

  // Import the Python module.
  pName = PyUnicode_DecodeFSDefault("plot");
  pModule = PyImport_Import(pName);

  // Get the reference to Python Function to call.
  pFunc = PyObject_GetAttrString(pModule, "clmplot");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(4);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyString_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 1, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 2, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 3, pFigHeight);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int histplot(const std::string& fname,
             const std::string& figTitle,
             const int figWidth = 6,
             const int figHeight = 4)
{

  // PyObject contains info Python needs to treat a pointer to an object as an object.
  // It contains object's reference count and pointer to corresponding object type.
  PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

  // Initialize Python Interpreter.
  Py_Initialize();
  // Import sys module in Interpreter and add current path to python search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");

  // Import the Python module.
  pName = PyUnicode_DecodeFSDefault("plot");
  pModule = PyImport_Import(pName);

  // Get the reference to Python Function to call.
  pFunc = PyObject_GetAttrString(pModule, "chistplot");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(4);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyString_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 1, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 2, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 3, pFigHeight);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

#endif