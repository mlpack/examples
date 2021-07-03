#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int impute(const std::string& fname,
           const std::string& kind = "mean",
           const std::string& dateCol = "")
{
    // Calls python function Imputer and fills the missing values using
    // the specified imputation policy and saves the dataset as .csv

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    pName = PyUnicode_DecodeFSDefault("preprocess");
    pModule = PyImport_Import(pName);

    pFunc = PyObject_GetAttrString(pModule, "cimputer");
    
    pArgs = PyTuple_New(3);
    
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    
    PyObject* pKind = PyString_FromString(kind.c_str());
    PyTuple_SetItem(pArgs, 1, pKind);
    
    PyObject* pDatecol = PyString_FromString(dateCol.c_str());
    PyTuple_SetItem(pArgs, 2, pDatecol);
    
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

int resample(const std::string& fname,
             const std::string& target,
             const std::string& negValue,
             const std::string& posValue,
             const std::string& kind,
             const std::string& dateCol = "",
             const int& randomState = 123)
             
{
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
    
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    pName = PyUnicode_DecodeFSDefault("preprocess");
    pModule = PyImport_Import(pName);

    pFunc = PyObject_GetAttrString(pModule, "cresample");
    
    pArgs = PyTuple_New(7);
    
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    
    PyObject* pTarget = PyString_FromString(target.c_str());
    PyTuple_SetItem(pArgs, 1, pTarget);
    
    PyObject* pNegValue = PyString_FromString(negValue.c_str());
    PyTuple_SetItem(pArgs, 2, pNegValue);
    
    PyObject* pPosValue = PyString_FromString(posValue.c_str());
    PyTuple_SetItem(pArgs, 3, pPosValue);
    
    PyObject* pKind = PyString_FromString(kind.c_str());
    PyTuple_SetItem(pArgs, 4, pKind);
    
    PyObject* pDateCol = PyString_FromString(dateCol.c_str());
    PyTuple_SetItem(pArgs, 5, pDateCol);
    
    PyObject* pRandState = PyLong_FromLong(randomState);
    PyTuple_SetItem(pArgs, 6, pRandState);
    
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

#endif
