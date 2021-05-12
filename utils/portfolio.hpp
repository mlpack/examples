#ifndef CPORTFOLIO_HPP
#define CPORTFOLIO_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int PortFolio(const std::string& stock0,
              const std::string& stock1,
              const std::string& stock2,
              const std::string& stock3,
              const std::string& start,
              const std::string& end,
              const std::string& filename = "output.csv")
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");
  PyRun_SimpleString("sys.path.append(\"/srv/conda/envs/notebook/include/\")");
  pName = PyUnicode_DecodeFSDefault("portfolio");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cportfolio");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(7);

      PyObject* pValueStock0 = PyUnicode_FromString(stock0.c_str());
      PyTuple_SetItem(pArgs, 0, pValueStock0);

      PyObject* pValueStock1 = PyUnicode_FromString(stock1.c_str());
      PyTuple_SetItem(pArgs, 1, pValueStock1);

      PyObject* pValueStock2 = PyUnicode_FromString(stock2.c_str());
      PyTuple_SetItem(pArgs, 2, pValueStock2);

      PyObject* pValueStock3 = PyUnicode_FromString(stock3.c_str());
      PyTuple_SetItem(pArgs, 3, pValueStock3);

      PyObject* pValueStart = PyUnicode_FromString(start.c_str());
      PyTuple_SetItem(pArgs, 4, pValueStart);

      PyObject* pValueEnd = PyUnicode_FromString(end.c_str());
      PyTuple_SetItem(pArgs, 5, pValueEnd);

      PyObject* pValueFilename = PyUnicode_FromString(filename.c_str());
      PyTuple_SetItem(pArgs, 6, pValueFilename);

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