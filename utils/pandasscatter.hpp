// Inside the C++ notebook we can use:
// PandasScatter("housing.csv", "longitude", "latitude", "output.png");
// auto im = xw::image_from_file("output.png").finalize();
// im

#ifndef C_PANDAS_SCATTER_C_PANDAS_SCATTER_HPP
#define C_PANDAS_SCATTER_C_PANDAS_SCATTER_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

// Here we use the same arguments as we used in the python script,
// since this is what is passed from the C++ notebook to call the python script.
int PandasScatter(const std::string& inFile,
                  const std::string& x,
                  const std::string& y,
                  const std::string& outFile = "output.png",
                  const int width = 10,
                  const int height = 10)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  // This has to be adapted if you run this on your local system,
  // so whenever you call the python script it can find the correct
  // module -> PYTHONPATH, on lab.mlpack.org we put all the utility
  // functions for plotting uinto the utils folder so we add that path
  // to the Python search path.
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  // Name of the python script without the extension.
  pName = PyUnicode_DecodeFSDefault("pandasscatter");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    // The Python function from the pandasscatter.py script
    // we like to call - cpandasscatter
    pFunc = PyObject_GetAttrString(pModule, "cpandasscatter");

    if (pFunc && PyCallable_Check(pFunc))
    {
      // The number of arguments we pass to the python script.
      // inFile, x, y, outFile='output.png', height=10, width=10
      // for the example above it's 6
      pArgs = PyTuple_New(6);

      // Now we have to encode the argument to the correct type
      // besides width, height everything else is a string.
      // So we can use PyUnicode_FromString.
      // If the data is an int we can use PyLong_FromLong,
      // see the lines below for an example.
      PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
      // Here we just set the index of the argument.
          PyTuple_SetItem(pArgs, 0, pValueinFile);

      PyObject* pValueX = PyUnicode_FromString(x.c_str());
      PyTuple_SetItem(pArgs, 1, pValueX);

      PyObject* pValueY = PyUnicode_FromString(y.c_str());
      PyTuple_SetItem(pArgs, 2, pValueY);

      PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
      PyTuple_SetItem(pArgs, 3, pValueoutFile);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 4, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 5, pValueHeight);

      // The rest of the c++ part can stay the same.

      pValue = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      if (pValue != NULL)
      {
        Py_DECREF(pValue);
      }
      else
      {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call failed.\n");
        return 1;
      }
    }
    else
    {
      if (PyErr_Occurred())
        PyErr_Print();
    }

    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else
  {
    PyErr_Print();
    return 1;
  }

  return 0;
}
int PandasScatterColor(const std::string& inFile,
                       const std::string& x,
                       const std::string& y,
                       const std::string& label,
                       const std::string& c,
                       const std::string& outFile,
                       const int width = 10,
                       const int height= 10)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("pandasscatter");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cpandasscattercolor");
    if( pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(8);

      PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
      PyTuple_SetItem(pArgs, 0, pValueinFile);

      PyObject* pValueX = PyUnicode_FromString(x.c_str());
      PyTuple_SetItem(pArgs, 1, pValueX);

      PyObject* pValueY = PyUnicode_FromString(y.c_str());
      PyTuple_SetItem(pArgs, 2, pValueY);

      PyObject* pValueLabel = PyUnicode_FromString(label.c_str());
      PyTuple_SetItem(pArgs, 3, pValueLabel);

       PyObject* pValueC = PyUnicode_FromString(c.c_str());
      PyTuple_SetItem(pArgs, 4, pValueC);

      PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
      PyTuple_SetItem(pArgs, 5, pValueoutFile);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 6, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 7, pValueHeight);

      pValue = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      if (pValue != NULL)
      {
        Py_DECREF(pValue);
      }
      else
      {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call Failed.\n");
        return 1;
      }
    }
    else
    {
      if (PyErr_Occurred())
        PyErr_Print();
    }

    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else
  {
    PyErr_Print();
    return -1;
  }
  return 0;
}
int PandasScatterMap(const std::string& inFile,
                     const std::string& imgFile,
                     const std::string& x,
                     const std::string& y,
                     const std::string& label,
                     const std::string& c,
                     const std::string& outFile,
                     const int width = 10,
                     const int height= 10)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("pandasscatter");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cpandasscattermap");
    if(pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(9);

      PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
      PyTuple_SetItem(pArgs, 0, pValueinFile);

      PyObject* pValueimgFile = PyUnicode_FromString(imgFile.c_str());
      PyTuple_SetItem(pArgs, 1, pValueimgFile);

      PyObject* pValueX = PyUnicode_FromString(x.c_str());
      PyTuple_SetItem(pArgs, 2, pValueX);

      PyObject* pValueY = PyUnicode_FromString(y.c_str());
      PyTuple_SetItem(pArgs, 3, pValueY);

      PyObject* pValueLabel = PyUnicode_FromString(label.c_str());
      PyTuple_SetItem(pArgs, 4, pValueLabel);

      PyObject* pValueC = PyUnicode_FromString(c.c_str());
      PyTuple_SetItem(pArgs, 5, pValueC);

      PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
      PyTuple_SetItem(pArgs, 6, pValueoutFile);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 7, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 8, pValueHeight);

      pValue = PyObject_CallObject(pFunc, pArgs);
      Py_DECREF(pArgs);
      if (pValue != NULL)
      {
        Py_DECREF(pValue);
      }
      else
      {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call Failed.\n");
        return 1;
      }

    }
    else
    {
      if (PyErr_Occurred())
        PyErr_Print();
    }

    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
  }
  else
  {
    PyErr_Print();
    return -1;
  }
  return 0;
}
#endif
