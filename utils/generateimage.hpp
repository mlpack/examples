// Inside C++ notebook we can use:
// GenerateImage("input.csv", "output.png")
// auto im = xw::image_from_file("output.png").finalize()
// im

#ifndef C_GENERATE_IMAGE_HPP
#define C_GENERATE_IMAGE_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

// Here we use the same arguments as we used in the python script,
// since this is what is passed from the C++ notebook to call the python script.
int GenerateImage(const std::string& inFile,
                  const std::string& outFile = "output.png")
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;

  // This has to be adapted if you run this on your local system,
  // so whenever you call the python script it can find the correct
  // module -> PYTHONPATH, on lab.mlpack.org we put all the utility
  // functions for plotting uinto the utils folder so we add that path
  // to the Python search path.

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  // Name of the python script without the extension.
  pName = PyUnicode_DecodeFSDefault("generateimage");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    // The Python function from the generateimage.py script
    // we like to call - cgenerateimage
    pFunc = PyObject_GetAttrString(pModule, "cgenerateimage");

    if (pFunc && PyCallable_Check(pFunc))
    {
      // The number of arguments we pass to the python script.
      // inFile, outFile='output.png'
      // for the example above it's 2
      pArgs = PyTuple_New(2);

      // Now we have to encode the argument to the correct type
      // besides width, height everything else is a string.
      // So we can use PyUnicode_FromString.
      // If the data is an int we can use PyLong_FromLong,
      // see the lines below for an example.
      PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
      // Here we just set the index of the argument.
      PyTuple_SetItem(pArgs, 0, pValueinFile);

      PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
      PyTuple_SetItem(pArgs, 1, pValueoutFile);

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
#endif