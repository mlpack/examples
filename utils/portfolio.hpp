#ifndef CPORTFOLIO_HPP
#define CPORTFOLIO_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int Portfolio(const std::string& stocks,
              const std::string& dataSource,
              const std::string& start,
              const std::string& end,
              const std::string& filePath = "portfolio.csv")
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("portfolio");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cportfolio");
    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(5);

      //! Comma separated stocks.
      PyObject* pValueStocks = PyUnicode_FromString(stocks.c_str());
      PyTuple_SetItem(pArgs, 0, pValueStocks);

      PyObject* pValueSource = PyUnicode_FromString(dataSource.c_str());
      PyTuple_SetItem(pArgs, 1, pValueSource);

      PyObject* pValueStart = PyUnicode_FromString(start.c_str());
      PyTuple_SetItem(pArgs, 2, pValueStart);

      PyObject* pValueEnd = PyUnicode_FromString(end.c_str());
      PyTuple_SetItem(pArgs, 3, pValueEnd);

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