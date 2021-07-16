#ifndef CSCATTER_CSCATTER_HPP
#define CSCATTER_CSCATTER_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int Scatter(const std::string& x,
            const std::string& y,
            const std::string& a,
            const std::string& c,
            const int size,
            const std::string& filename = "output.gif",
            const int width = 2000,
            const int height = 4000)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("scatter");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cscatter");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(8);

      PyObject* pValueX = PyUnicode_FromString(x.c_str());
      PyTuple_SetItem(pArgs, 0, pValueX);

      PyObject* pValueY = PyUnicode_FromString(y.c_str());
      PyTuple_SetItem(pArgs, 1, pValueY);

      PyObject* pValueA = PyUnicode_FromString(a.c_str());
      PyTuple_SetItem(pArgs, 2, pValueA);

      PyObject* pValueC = PyUnicode_FromString(c.c_str());
      PyTuple_SetItem(pArgs, 3, pValueC);

      PyObject* pValueSize = PyLong_FromLong(size);
      PyTuple_SetItem(pArgs, 4, pValueSize);

      PyObject* pValueFilename = PyUnicode_FromString(filename.c_str());
      PyTuple_SetItem(pArgs, 5, pValueFilename);

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
