#ifndef PREPROCESS_HPP
#define PREPROCESS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int Impute(const std::string& fname,
           const std::string& kind = "mean",
           const std::string& dateCol = "",
           const std::string& dataDir = "data")
{
    // Calls python function Imputer and fills the missing values using
    // the specified imputation policy and saves the dataset as .csv
    
    // PyObject contains info Python needs to treat a pointer to an object as an object.
    // It contains object's reference count and pointer to corresponding object type.
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    // Initialize Python Interpreter.
    Py_Initialize();
    // Import sys module in Interpreter and add current path to python search path.
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    // Import the Python module.
    pName = PyUnicode_DecodeFSDefault("preprocess");
    pModule = PyImport_Import(pName);

    // Get the reference to Python Function to call.
    pFunc = PyObject_GetAttrString(pModule, "cimputer");
    
    // Create a tuple object to hold the arguments for function call.
    pArgs = PyTuple_New(4);
    
    // String object representing the name of the dataset to be loaded.
    PyObject* pFname = PyUnicode_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    
    // String object representing the type of method used for imputation, defaults to mean.
    PyObject* pKind = PyUnicode_FromString(kind.c_str());
    PyTuple_SetItem(pArgs, 1, pKind);
    
    // String object representing the name of the feature to be parsed as TimeStamp.
    PyObject* pDatecol = PyUnicode_FromString(dateCol.c_str());
    PyTuple_SetItem(pArgs, 2, pDatecol);
    
    // String object representing the directory in which Data is saved.
    PyObject* pDataDir = PyUnicode_FromString(dataDir.c_str());
    PyTuple_SetItem(pArgs, 3, pDataDir);
    
    // Call the function by passing the reference to function & tuple holding arguments.
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

int Resample(const std::string& fname,
             const std::string& target,
             const std::string& negValue,
             const std::string& posValue,
             const std::string& kind,
             const std::string& dateCol = "",
             const int& randomState = 123,
             const std::string& dataDir = "data")
             
{
    // Calls python function Resample to resample the target classes using
    // the specified resamling method and saves the dataset as .csv
    
    // PyObject contains info Python needs to treat a pointer to an object as an object.
    // It contains object's reference count and pointer to corresponding object type.
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
    
    // Initialize Python Interpreter.
    Py_Initialize();
    // Import sys module in Interpreter and add current path to python search path.
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    // Import the Python module.
    pName = PyUnicode_DecodeFSDefault("preprocess");
    pModule = PyImport_Import(pName);

    // Get the reference to Python Function to call.
    pFunc = PyObject_GetAttrString(pModule, "cresample");
    
    // Create a tuple object to hold the arguments for function call.
    pArgs = PyTuple_New(8);
    
    // String object representing the name of the dataset to be loaded.
    PyObject* pFname = PyUnicode_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    
    // String object representing the target column.
    PyObject* pTarget = PyUnicode_FromString(target.c_str());
    PyTuple_SetItem(pArgs, 1, pTarget);
    
    // String object representing the negative class.
    PyObject* pNegValue = PyUnicode_FromString(negValue.c_str());
    PyTuple_SetItem(pArgs, 2, pNegValue);
    
    // String object representing the positive class.
    PyObject* pPosValue = PyUnicode_FromString(posValue.c_str());
    PyTuple_SetItem(pArgs, 3, pPosValue);
    
    // String object representing the kind of resampling.
    PyObject* pKind = PyUnicode_FromString(kind.c_str());
    PyTuple_SetItem(pArgs, 4, pKind);
    
    // String object representing the name of the feature to be parsed as TimeStamp.
    PyObject* pDateCol = PyUnicode_FromString(dateCol.c_str());
    PyTuple_SetItem(pArgs, 5, pDateCol);
    
    // Integer object for random state.
    PyObject* pRandState = PyLong_FromLong(randomState);
    PyTuple_SetItem(pArgs, 6, pRandState);
    
    // String object representing the directory in which Data is saved.
    PyObject* pDataDir = PyUnicode_FromString(dataDir.c_str());
    PyTuple_SetItem(pArgs, 7, pDataDir);
    
    // Call the function by passing the reference to function & tuple holding arguments.
    pValue = PyObject_CallObject(pFunc, pArgs);
    
    return 0;
}

int Resample(const std::string& fname,
             const std::string& target,
             const int negValue = 0,
             const int posValue = 1,
             const std::string& kind = "oversample",
             const std::string& dateCol = "",
             const int& randomState = 123,
             const std::string& dataDir = "data")

{
    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");

    pName = PyUnicode_DecodeFSDefault("preprocess");
    pModule = PyImport_Import(pName);

    pFunc = PyObject_GetAttrString(pModule, "cresamplenum");

    pArgs = PyTuple_New(8);

    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);

    PyObject* pTarget = PyString_FromString(target.c_str());
    PyTuple_SetItem(pArgs, 1, pTarget);

    PyObject* pNegValue = PyLong_FromLong(negValue);
    PyTuple_SetItem(pArgs, 2, pNegValue);

    PyObject* pPosValue = PyLong_FromLong(posValue);
    PyTuple_SetItem(pArgs, 3, pPosValue);

    PyObject* pKind = PyString_FromString(kind.c_str());
    PyTuple_SetItem(pArgs, 4, pKind);

    PyObject* pDateCol = PyString_FromString(dateCol.c_str());
    PyTuple_SetItem(pArgs, 5, pDateCol);

    PyObject* pRandState = PyLong_FromLong(randomState);
    PyTuple_SetItem(pArgs, 6, pRandState);
    
    // String object representing the directory in which Data is saved.
    PyObject* pDataDir = PyUnicode_FromString(dataDir.c_str());
    PyTuple_SetItem(pArgs, 7, pDataDir);

    pValue = PyObject_CallObject(pFunc, pArgs);

    return 0;
}

#endif

