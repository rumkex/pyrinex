#include "library.h"

#include <numpy/arrayobject.h>
#include <stdio.h>
#include "py_generator.h"
#include "py_record.h"
#include "py_sp3.h"

static PyMethodDef methods[] = {
        {"load", (PyCFunction)rinex_load, METH_VARARGS | METH_KEYWORDS, "Spam."},
        {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "rinex",
        NULL,
        -1,
        methods
};

static PyObject * rinex_load(PyObject *self, PyObject *args, PyObject *kwargs) {
    char* keywords[] = { "path", "meas", "svs", NULL };
    PyObject *path = NULL, *meas = NULL, *svs = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", keywords, &path, &meas, &svs))
    {
        return NULL;
    }

    if (meas != Py_None && meas != NULL && !PySequence_Check(meas)) {
        PyErr_SetString(PyExc_ValueError, "meas must be a sequence or None");
        return NULL;
    }

    if (svs != Py_None && svs != NULL && !PySequence_Check(svs)) {
        PyErr_SetString(PyExc_ValueError, "svs must be a sequence or None");
        return NULL;
    }

    PyObject *gen_args = Py_BuildValue("(OOO)", path, meas ? meas: Py_None, svs ? svs: Py_None);

    if (!gen_args) {
        return NULL;
    }

    PyObject* gen = PyObject_CallObject((PyObject *) &RinexGeneratorType, gen_args);

    Py_DECREF(gen_args);

    return gen;
}


__attribute_used__
PyMODINIT_FUNC PyInit__rinex(void) {
    import_array();

    PyObject* m = PyModule_Create(&module);

    m = init_record_type(m);
    m = init_generator_type(m);
    m = init_ephemeris_type(m);

    return m;
}
