#ifndef LIBRINEX_PY_SP3_H
#define LIBRINEX_PY_SP3_H

#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "config.h"

typedef struct {
    PyObject_HEAD;
    PyObject *start_time;
    PyObject *records;
    PyObject *epoches;
    PyObject *sats;
} Ephemeris;

extern PyTypeObject EphemerisType;

PyObject *Ephemeris_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Ephemeris_init(Ephemeris *self, PyObject *args);
void Ephemeris_dealloc(Ephemeris *self);

PyObject *init_ephemeris_type(PyObject *m);

#endif //LIBRINEX_PY_SP3_H
