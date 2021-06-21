#ifndef PLOT_HPP
#define PLOT_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>

int scatter(const std::string fname, const std::string type, const int figWidth = 26, const int figHeight = 7) {

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../utils/\")");
    pName = PyUnicode_DecodeFSDefault("plot");
    pModule = PyImport_Import(pName);
    //Py_DECREF(pName);
    
    pFunc = PyObject_GetAttrString(pModule, "cscatter");
    pArgs = PyTuple_New(4);
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    PyObject* pType = PyString_FromString(type.c_str());
    PyTuple_SetItem(pArgs, 1, pType);
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 2, pFigWidth);
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 3, pFigHeight);
    pValue = PyObject_CallObject(pFunc, pArgs);

    /*
    if(pModule != NULL) {
        
        if(pFunc && PyCallable_Check(pFunc)) {
            std::cout << "Callable Method Found :)" << std::endl;
    */
            
            
//             /*PyObject* pXdim = PyLong_FromLong(xdim);
//             PyTuple_SetItem(pArgs, 1, pXdim);
//             PyObject* pYdim = PyLong_FromLong(ydim);
//             PyTuple_SetItem(pArgs, 2, pYdim);
//             */
            
            
//             Py_DECREF(pArgs);
//             if(pValue != NULL) {
//                 Py_DECREF(pValue);
//             }else{
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr, "Call failed\n");
//                 return 1;
//             }
    /*
        }else{
            if(PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }else{
        PyErr_Print();
        return 1;
    }
        */
    return 0;
}

int barplot(const std::string fname, const std::string x, const std::string y, const std::string figTitle, const int figWidth = 5, const int figHeight = 7) {

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./\")");
    pName = PyUnicode_DecodeFSDefault("plot");
    pModule = PyImport_Import(pName);
    //Py_DECREF(pName);
    
    pFunc = PyObject_GetAttrString(pModule, "cbarplot");
    pArgs = PyTuple_New(6);
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    PyObject* pX = PyString_FromString(x.c_str());
    PyTuple_SetItem(pArgs, 1, pX);
    PyObject* pY = PyString_FromString(y.c_str());
    PyTuple_SetItem(pArgs, 2, pY);
    PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
    PyTuple_SetItem(pArgs, 3, pFigTitle);
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 4, pFigWidth);
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 5, pFigHeight);
    pValue = PyObject_CallObject(pFunc, pArgs);

    /*
    if(pModule != NULL) {
        
        if(pFunc && PyCallable_Check(pFunc)) {
            std::cout << "Callable Method Found :)" << std::endl;
    */
            
            
//             /*PyObject* pXdim = PyLong_FromLong(xdim);
//             PyTuple_SetItem(pArgs, 1, pXdim);
//             PyObject* pYdim = PyLong_FromLong(ydim);
//             PyTuple_SetItem(pArgs, 2, pYdim);
//             */
            
            
//             Py_DECREF(pArgs);
//             if(pValue != NULL) {
//                 Py_DECREF(pValue);
//             }else{
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr, "Call failed\n");
//                 return 1;
//             }
    /*
        }else{
            if(PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }else{
        PyErr_Print();
        return 1;
    }
        */
    return 0;
}

int heatmap(const std::string fname, const std::string colorMap, const std::string figTitle, const int annotation = false, const int figWidth = 12, const int figHeight = 6) {

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./\")");
    pName = PyUnicode_DecodeFSDefault("plot");
    pModule = PyImport_Import(pName);
    //Py_DECREF(pName);
    
    pFunc = PyObject_GetAttrString(pModule, "cheatmap");
    pArgs = PyTuple_New(6);
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    PyObject* pColorMap = PyString_FromString(colorMap.c_str());
    PyTuple_SetItem(pArgs, 1, pColorMap);
    PyObject* pAnnotation = PyBool_FromLong(annotation);
    PyTuple_SetItem(pArgs, 2, pAnnotation);
    PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
    PyTuple_SetItem(pArgs, 3, pFigTitle);
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 4, pFigWidth);
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 5, pFigHeight);
    pValue = PyObject_CallObject(pFunc, pArgs);

    /*
    if(pModule != NULL) {
        
        if(pFunc && PyCallable_Check(pFunc)) {
            std::cout << "Callable Method Found :)" << std::endl;
    */
            
            
//             /*PyObject* pXdim = PyLong_FromLong(xdim);
//             PyTuple_SetItem(pArgs, 1, pXdim);
//             PyObject* pYdim = PyLong_FromLong(ydim);
//             PyTuple_SetItem(pArgs, 2, pYdim);
//             */
            
            
//             Py_DECREF(pArgs);
//             if(pValue != NULL) {
//                 Py_DECREF(pValue);
//             }else{
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr, "Call failed\n");
//                 return 1;
//             }
    /*
        }else{
            if(PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }else{
        PyErr_Print();
        return 1;
    }
        */
    return 0;
}

int lmplot(const std::string fname, const int figWidth = 6, const int figHeight = 7) {

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./\")");
    pName = PyUnicode_DecodeFSDefault("plot");
    pModule = PyImport_Import(pName);
    //Py_DECREF(pName);
    
    pFunc = PyObject_GetAttrString(pModule, "clmplot");
    pArgs = PyTuple_New(3);
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 1, pFigWidth);
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 2, pFigHeight);
    pValue = PyObject_CallObject(pFunc, pArgs);

    /*
    if(pModule != NULL) {
        
        if(pFunc && PyCallable_Check(pFunc)) {
            std::cout << "Callable Method Found :)" << std::endl;
    */
            
            
//             /*PyObject* pXdim = PyLong_FromLong(xdim);
//             PyTuple_SetItem(pArgs, 1, pXdim);
//             PyObject* pYdim = PyLong_FromLong(ydim);
//             PyTuple_SetItem(pArgs, 2, pYdim);
//             */
            
            
//             Py_DECREF(pArgs);
//             if(pValue != NULL) {
//                 Py_DECREF(pValue);
//             }else{
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr, "Call failed\n");
//                 return 1;
//             }
    /*
        }else{
            if(PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }else{
        PyErr_Print();
        return 1;
    }
        */
    return 0;
}

int histplot(const std::string fname, const std::string figTitle, const int figWidth = 6, const int figHeight = 4) {

    PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./\")");
    pName = PyUnicode_DecodeFSDefault("plot");
    pModule = PyImport_Import(pName);
    //Py_DECREF(pName);
    
    pFunc = PyObject_GetAttrString(pModule, "chistplot");
    pArgs = PyTuple_New(4);
    PyObject* pFname = PyString_FromString(fname.c_str());
    PyTuple_SetItem(pArgs, 0, pFname);
    PyObject* pFigTitle = PyString_FromString(figTitle.c_str());
    PyTuple_SetItem(pArgs, 1, pFigTitle);
    PyObject* pFigWidth = PyLong_FromLong(figWidth);
    PyTuple_SetItem(pArgs, 2, pFigWidth);
    PyObject* pFigHeight = PyLong_FromLong(figHeight);
    PyTuple_SetItem(pArgs, 3, pFigHeight);
    pValue = PyObject_CallObject(pFunc, pArgs);

    /*
    if(pModule != NULL) {
        
        if(pFunc && PyCallable_Check(pFunc)) {
            std::cout << "Callable Method Found :)" << std::endl;
    */
            
            
//             /*PyObject* pXdim = PyLong_FromLong(xdim);
//             PyTuple_SetItem(pArgs, 1, pXdim);
//             PyObject* pYdim = PyLong_FromLong(ydim);
//             PyTuple_SetItem(pArgs, 2, pYdim);
//             */
            
            
//             Py_DECREF(pArgs);
//             if(pValue != NULL) {
//                 Py_DECREF(pValue);
//             }else{
//                 Py_DECREF(pFunc);
//                 Py_DECREF(pModule);
//                 PyErr_Print();
//                 fprintf(stderr, "Call failed\n");
//                 return 1;
//             }
    /*
        }else{
            if(PyErr_Occurred())
                PyErr_Print();
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }else{
        PyErr_Print();
        return 1;
    }
        */
    return 0;
}

#endif

