#ifndef CPLOT_3D_HPP
#define CPLOT_3D_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int Plot3D(const std::string& x,
     const std::string& y,
     const std::string& z,
     const std::string& label,
     const std::string& xAxisLabel,
     const std::string& yAxisLabel,
     const std::string& zAxisLabel,
     const int mode = 2,
     const std::string& filename = "output.png",
     const int width = 10,
     const int height = 10)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("plot3d");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cplot3d");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(11);

      PyObject* pValueX = PyUnicode_FromString(x.c_str());
      PyTuple_SetItem(pArgs, 0, pValueX);

      PyObject* pValueY = PyUnicode_FromString(y.c_str());
      PyTuple_SetItem(pArgs, 1, pValueY);

      PyObject* pValueZ = PyUnicode_FromString(z.c_str());
      PyTuple_SetItem(pArgs, 2, pValueZ);

      PyObject* pValueLabel = PyUnicode_FromString(label.c_str());
      PyTuple_SetItem(pArgs, 3, pValueLabel);

      PyObject* pValuexAxisLabel = PyUnicode_FromString(xAxisLabel.c_str());
      PyTuple_SetItem(pArgs, 4, pValuexAxisLabel);

      PyObject* pValueyAxisLabel = PyUnicode_FromString(yAxisLabel.c_str());
      PyTuple_SetItem(pArgs, 5, pValueyAxisLabel);

      PyObject* pValuezAxisLabel = PyUnicode_FromString(zAxisLabel.c_str());
      PyTuple_SetItem(pArgs, 6, pValuezAxisLabel);

      PyObject* pValueMode = PyLong_FromLong(mode);
      PyTuple_SetItem(pArgs, 7, pValueMode);

      PyObject* pValueFilename = PyUnicode_FromString(filename.c_str());
      PyTuple_SetItem(pArgs, 8, pValueFilename);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 9, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 10, pValueHeight);

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
