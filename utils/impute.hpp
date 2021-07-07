// Inside the C++ notebook we can use:
// Impute("filename.csv", "output.csv", "imputationMethod")
// imputationMethod can be "mean", "median", "method" depending upon missing values.

#ifndef CIMPUTE_HPP
#define CIMPUTE_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

// Here, we will use the same argument as used in python script impute.py
// since this is what passed from the C++ notebook to python script.

int Impute(const std::string& inFile,
           const std::string& outFile,
           const std::string& kind)
{
    // Calls python function Imputer and fills the missing values using
    // the specified imputation policy and saves the dataset as .csv.
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

    // Name of python script without extension.
    pName = PyUnicode_DecodeFSDefault("impute");

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL)
    {
        // The Python function from the impute.py script
        // we like to call - cimputer
        pFunc = PyObject_GetAttrString(pModule, "cimputer");

        if(pFunc && PyCallable_Check(pFunc))
        {
            // The number of arguments we pass to the python script.
            // inFile, outFile, kind
            // for the function above it's 3
            pArgs = PyTuple_New(3);

            // Now we have to encode the argument to the correct type
            // besides width , height everything else is a string.
            // So we can use PyUnicode_FromString.

            PyObject* pValueinFile = PyUnicode_FromString(inFile.c_str());
            //Here we just set the index of the argument.
            PyTuple_SetItem(pArgs, 0, pValueinFile);

            PyObject* pValueoutFile = PyUnicode_FromString(outFile.c_str());
            PyTuple_SetItem(pArgs, 1, pValueoutFile);

            PyObject* pValuekind = PyUnicode_FromString(kind.c_str());
            PyTuple_SetItem(pArgs, 2, pValuekind);

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
