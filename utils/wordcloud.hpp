#ifndef CWORD_CLOUD_HPP
#define CWORD_CLOUD_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int WordCloud(const std::string& words,
       const std::string& filename = "output.png",
       const int width = 2000,
       const int height = 4000)
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue;
  int i;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\"../utils/\")");
  pName = PyUnicode_DecodeFSDefault("wordcloud");

  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL)
  {
    pFunc = PyObject_GetAttrString(pModule, "cwordcloud");

    if (pFunc && PyCallable_Check(pFunc))
    {
      pArgs = PyTuple_New(4);

      PyObject* pValueWords = PyUnicode_FromString(words.c_str());
      PyTuple_SetItem(pArgs, 0, pValueWords);

      PyObject* pValueFilename = PyUnicode_FromString(filename.c_str());
      PyTuple_SetItem(pArgs, 1, pValueFilename);

      PyObject* pValueWidth = PyLong_FromLong(width);
      PyTuple_SetItem(pArgs, 2, pValueWidth);

      PyObject* pValueHeight = PyLong_FromLong(height);
      PyTuple_SetItem(pArgs, 3, pValueHeight);

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
