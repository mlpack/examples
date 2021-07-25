// Inside the C++ notebook we can use:
// Hist("filename.csv", "bins", "histogram.png")

#ifndef CHISTOGRAM_HPP
#define CHISTOGRAM_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

// Here, we will use the same argument as used in python script histogram.py
// since this is what passed from the C++ notebook to python script

int Hist(const std::string& inFile,
         const int bins,
         const int width = 20,
         const int height = 15,
         const std::string& outFile = "histogram.png")

{
    // Calls python function cpandahist and plots the histogram

    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;

    // This has to be adapted if you run this on your local system,
    // so whenever you call the python script it can find the correct
    // module -> PYTHONPATH, on lab.mlpack.org we put all the utility
    // functions in the utils folder so we add that path
    // to the Python search path.
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    // Name of python script without extension
    pName = PyUnicode_DecodeFSDefault("histogram");

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL)
    {
        // The Python function from the histogram.py script
        // we like to call - cpandashist
        pFunc = PyObject_GetAttrString(pModule, "cpandashist");

        if (pFunc && PyCallable_Check(pFunc))
        {
            // The number of arguments we pass to the python script.
            // inFile, outFile, kind
            // for the function above it's 5
            pArgs = PyTuple_New(5);

            // Now we have to encode the argument to the correct type
            // We can use PyLong_FromLong for bins, width and height as they are integers
            // As for rest, we can use PyString_FromString

            PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
            //Here we just set the index of the argument.
            PyTuple_SetItem(pArgs, 0, pValueinFile);

            PyObject* pValuebins = PyLong_FromLong(bins);
            PyTuple_SetItem(pArgs, 1, pValuebins);

            PyObject* pValuewidth = PyLong_FromLong(width);
            PyTuple_SetItem(pArgs, 2, pValuewidth);

            PyObject* pValueheight = PyLong_FromLong(height);
            PyTuple_SetItem(pArgs, 3, pValueheight);

            PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
            PyTuple_SetItem(pArgs, 4, pValueoutFile);

            // The rest of the c++ part can remain same.

            pValue = PyObject_CallObject(pFunc, pArgs);
            // We call the object with function name and arguments provided in c++ notebook.
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
            return -1;
        }
        return 0;
    }

#endif
