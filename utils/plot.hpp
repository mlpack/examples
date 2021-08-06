#ifndef PLOT_HPP
#define PLOT_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int ScatterPlot(const std::string& fname,
                const std::string& xCol,
                const std::string& yCol,
                const std::string& dateCol = "",
                const std::string& maskCol = "",
                const std::string& type = "",
                const std::string& color = "",
                const std::string& xLabel = "",
                const std::string& yLabel = "",
                const std::string& figTitle = "",
                const int figWidth = 26,
                const int figHeight = 7,
                const std::string& plotDir = "plots")
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
  pArgs = PyTuple_New(13);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature to be plotted along X axis.
  PyObject* pXcol = PyUnicode_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXcol);

  // String object representing the name of the feature to be plotted along Y axis.
  PyObject* pYcol = PyUnicode_FromString(yCol.c_str());
  PyTuple_SetItem(pArgs, 2, pYcol);

  // String object representing the name of the feature to be parsed as TimeStamp.
  PyObject* pDateCol = PyUnicode_FromString(dateCol.c_str());
  PyTuple_SetItem(pArgs, 3, pDateCol);

  // String object representing the name of the feature to be used to mask the plot data points.
  PyObject* pMaskCol = PyUnicode_FromString(maskCol.c_str());
  PyTuple_SetItem(pArgs, 4, pMaskCol);

  // String object representing the value for masking.
  PyObject* pType = PyUnicode_FromString(type.c_str());
  PyTuple_SetItem(pArgs, 5, pType);

  // String object representing the feature name to be used as color value in plot.
  PyObject* pColor = PyUnicode_FromString(color.c_str());
  PyTuple_SetItem(pArgs, 6, pColor);

  // String object representing the X axis label.
  PyObject* pXlabel = PyUnicode_FromString(xLabel.c_str());
  PyTuple_SetItem(pArgs, 7, pXlabel);

  // String object representing the Y axis label.
  PyObject* pYlabel = PyUnicode_FromString(yLabel.c_str());
  PyTuple_SetItem(pArgs, 8, pYlabel);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 9, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 10, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 11, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 12, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int BarPlot(const std::string& fname,
            const std::string& x,
            const std::string& y,
            const std::string& dateCol = "",
            const std::string& figTitle = "",
            const int figWidth = 5,
            const int figHeight = 7,
            const std::string& plotDir = "plots")
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
  pArgs = PyTuple_New(8);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature to be plotted along X axis.
  PyObject* pX = PyUnicode_FromString(x.c_str());
  PyTuple_SetItem(pArgs, 1, pX);

  // String object representing the name of the feature to be plotted along Y axis.
  PyObject* pY = PyUnicode_FromString(y.c_str());
  PyTuple_SetItem(pArgs, 2, pY);

  // String object representing the name of the feature to be parsed as TimeStamp.
  PyObject* pDateCol = PyUnicode_FromString(dateCol.c_str());
  PyTuple_SetItem(pArgs, 3, pDateCol);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 4, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 5, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 6, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 7, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int HeatMapPlot(const std::string& fname,
                const std::string& colorMap,
                const std::string& figTitle = "",
                const int annotation = false,
                const int figWidth = 15,
                const int figHeight = 15,
                const std::string& plotDir = "plots")
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
  pArgs = PyTuple_New(7);

  // String object representing the name of the dataset to be loaded. 
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of color map to be used for plotting.
  PyObject* pColorMap = PyUnicode_FromString(colorMap.c_str());
  PyTuple_SetItem(pArgs, 1, pColorMap);

  // Boolean object indicating if correlation values must be annotated in figure.
  PyObject* pAnnotation = PyBool_FromLong(annotation);
  PyTuple_SetItem(pArgs, 2, pAnnotation);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 3, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 4, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 5, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 6, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int LmPlot(const std::string& fname,
           const std::string& xCol,
           const std::string& yCol,
           const std::string& figTitle,
           const int figWidth = 6,
           const int figHeight = 7,
           const std::string& plotDir = "plots")
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
  pArgs = PyTuple_New(7);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);
    
  // String object representing the name of the X (feature) column.
  PyObject* pXCol = PyUnicode_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXCol);
    
  // String object representing the name of the y (target) column.
  PyObject* pYCol = PyUnicode_FromString(yCol.c_str());
  PyTuple_SetItem(pArgs, 2, pYCol);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 3, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 4, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 5, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 6, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int HistPlot(const std::string& fname,
             const std::string& xCol,
             const std::string& figTitle,
             const int figWidth = 6,
             const int figHeight = 4,
             const std::string& plotDir = "plots")
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
  pArgs = PyTuple_New(6);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);
    
  // String object representing the name of the X (feature) column.
  PyObject* pXCol = PyUnicode_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXCol);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 2, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 3, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 4, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 5, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int MissingPlot(const std::string& fname,
                const std::string& colorMap,
                const std::string& figTitle = "",
                const int figWidth = 6,
                const int figHeight = 4,
                const std::string& plotDir = "plots")
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
  pFunc = PyObject_GetAttrString(pModule, "cmissing");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(6);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of color map to be used for plotting.
  PyObject* pColorMap = PyUnicode_FromString(colorMap.c_str());
  PyTuple_SetItem(pArgs, 1, pColorMap);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 2, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 3, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 4, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 5, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);

  return 0;
}

int CountPlot(const std::string& fname,
              const std::string& xCol,
              const std::string& figTitle = "",
              const std::string& hue = "",
              const int figWidth = 6,
              const int figHeight = 4,
              const std::string& plotDir = "plots")
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
  pFunc = PyObject_GetAttrString(pModule, "ccountplot");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(7);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature to count for.
  PyObject* pXcol = PyUnicode_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXcol);

  // String object representing the feature to be used as hue. 
  PyObject* pHue = PyUnicode_FromString(hue.c_str());
  PyTuple_SetItem(pArgs, 2, pHue);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 3, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 4, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 5, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 6, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);
    
  return 0;

}

int RocAucPlot(const std::string& yTrue,
               const std::string& probs,
               const std::string& figTitle = "roc_auc",
               const std::string& plotDir = "plots")
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
    pFunc = PyObject_GetAttrString(pModule, "cplotRocAUC");
    
    // Create a tuple object to hold the arguments for function call.
    pArgs = PyTuple_New(4);
    
    // String object representing the filename of the csv file containing targets.
    PyObject* pYtrue = PyUnicode_FromString(yTrue.c_str());
    PyTuple_SetItem(pArgs, 0, pYtrue);
    
    // String object representing the filename of the csv file containing the probabilities.
    PyObject* pProbs = PyUnicode_FromString(probs.c_str());
    PyTuple_SetItem(pArgs, 1, pProbs);
    
    // String object representing the title of the figure.
    PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
    PyTuple_SetItem(pArgs, 2, pFigTitle);
    
    // String object representing the directory in which plot is saved.
    PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
    PyTuple_SetItem(pArgs, 3, pPlotDir);
    
    // Call the function by passing the reference to function & tuple holding arguments.
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

int LinePlot(const std::string& fname,
             const std::string& xCol,
             const std::string& yCol,
             const std::string& figTitle = "",
             const int figWidth = 16,
             const int figHeight = 6,
             const std::string& plotDir = "plots")
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
  pFunc = PyObject_GetAttrString(pModule, "clineplot");

  // Create a tuple object to hold the arguments for function call.
  pArgs = PyTuple_New(7);

  // String object representing the name of the dataset to be loaded.
  PyObject* pFname = PyUnicode_FromString(fname.c_str());
  PyTuple_SetItem(pArgs, 0, pFname);

  // String object representing the name of the feature (X).
  PyObject* pXcol = PyUnicode_FromString(xCol.c_str());
  PyTuple_SetItem(pArgs, 1, pXcol);

  // String object representing the name of the target (Y).
  PyObject* pYcol = PyUnicode_FromString(yCol.c_str());
  PyTuple_SetItem(pArgs, 2, pYcol);

  // String object representing the title of the figure.
  PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
  PyTuple_SetItem(pArgs, 3, pFigTitle);

  // Integer object representing the width of the figure.
  PyObject* pFigWidth = PyLong_FromLong(figWidth);
  PyTuple_SetItem(pArgs, 4, pFigWidth);

  // Integer object representing the height of the figure.
  PyObject* pFigHeight = PyLong_FromLong(figHeight);
  PyTuple_SetItem(pArgs, 5, pFigHeight);
    
  // String object representing the directory in which plot is saved.
  PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
  PyTuple_SetItem(pArgs, 6, pPlotDir);

  // Call the function by passing the reference to function & tuple holding arguments.
  pValue = PyObject_CallObject(pFunc, pArgs);
    
  return 0;

}

int PlotCatData(const std::string& fName,
                const int targetCol,
                const std::string& xLabel,
                const std::string& yLabel,
                const std::string& figTitle = "",
                const int figWidth = 8,
                const int figHeight = 6,
                const std::string& plotDir = "plots")
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
    pFunc = PyObject_GetAttrString(pModule, "cplotCatData");
    
    // Create a tuple object to hold the arguments for function call.
    pArgs = PyTuple_New(8);
    
    // String object representing the name of the dataset to be loaded.
    PyObject* pFname = PyUnicode_FromString(fName.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    
    // Integer object representing the target column
    PyObject* pTargetCol = PyLong_FromLong(targetCol);
    PyTuple_SetItem(pArgs, 1, pTargetCol);

    // String object representing the X-Label.
    PyObject* pXlabel = PyUnicode_FromString(xLabel.c_str());
    PyTuple_SetItem(pArgs, 2, pXlabel);

    // String object representing the Y-Label. 
    PyObject* pYlabel = PyUnicode_FromString(yLabel.c_str());
    PyTuple_SetItem(pArgs, 3, pYlabel);

    // String object representing the title of the figure.
    PyObject* pFigTitle = PyUnicode_FromString(figTitle.c_str());
    PyTuple_SetItem(pArgs, 4, pFigTitle);

    // Integer object representing the width of the figure.
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 5, pFigWidth);

    // Integer object representing the height of the figure.
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 6, pFigHeight);
    
    // String object representing the directory in which plot is saved.
    PyObject* pPlotDir = PyUnicode_FromString(plotDir.c_str());
    PyTuple_SetItem(pArgs, 7, pPlotDir);
    
    // Call the function by passing the reference to function & tuple holding arguments.
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

#endif
