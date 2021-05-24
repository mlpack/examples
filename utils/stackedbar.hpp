#ifndef CSTACKED_BAR_HPP
#define CSTACKED_BAR_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int StackedBar(const std::string& values,
       const std::string& colors,
       const std::string& filename = "output.png",
       const int width = 3,
       const int height = 10)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("stackedbar");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cstackedbar");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(5);

      PyObject* pValueValues = PyUnicode_FromString(values.c_str());
      PyTuple_SetItem(pArgs, 0, pValueValues);

      PyObject* pValueColors= PyUnicode_FromString(colors.c_str());
      PyTuple_SetItem(pArgs, 1, pValueColors);

      PyObject* pValueFilename = PyUnicode_FromString(filename.c_str());
      PyTuple_SetItem(pArgs, 2, pValueFilename);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 3, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 4, pValueHeight);

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
        fprintf(stderr,"Call failed\n");
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
