#ifndef CFRONT_HPP
#define CFRONT_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int Front(const std::string& nsga2DataX,
          const std::string& nsga2DataY,
          const std::string& moeadDataX,
          const std::string& moeadDataY,
          const std::string& filePath = "fronts.gif")
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("front");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cfront");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(5);

      PyObject* pValueNSGA2X = PyUnicode_FromString(nsga2DataX.c_str());
      PyTuple_SetItem(pArgs, 0, pValueNSGA2X);

      PyObject* pValueNSGA2Y = PyUnicode_FromString(nsga2DataY.c_str());
      PyTuple_SetItem(pArgs, 1, pValueNSGA2Y);

      PyObject* pValueMOEADX = PyUnicode_FromString(moeadDataX.c_str());
      PyTuple_SetItem(pArgs, 2, pValueMOEADX);

      PyObject* pValueMOEADY = PyUnicode_FromString(moeadDataY.c_str());
      PyTuple_SetItem(pArgs, 3, pValueMOEADY);

      PyObject* pValueFilePath = PyUnicode_FromString(filePath.c_str());
      PyTuple_SetItem(pArgs, 4, pValueFilePath);

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