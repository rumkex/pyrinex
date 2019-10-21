#ifndef LIBRINEX_PY_RECORD_H
#define LIBRINEX_PY_RECORD_H

#include "config.h"
#include "py_generator.h"
#include <Python.h>


typedef struct rinex_value {
    int present;
    long value;
    short ssi;
    short lli;
} rinex_value;

typedef struct rinex_row {
    int satSys;
    int satId;
    rinex_value values[MAX_OBS_TYPES];
} rinex_row;

typedef struct {
    PyObject_HEAD
    PyObject *header;
    PyObject* time;
    rinex_row rows[MAX_SATS];

} RinexRecord;


extern PyTypeObject RinexRecordType;

PyObject* init_record_type(PyObject* m);

void Record_dealloc(RinexRecord* self);
PyObject * Record_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Record_init(RinexRecord *self, PyObject *args, PyObject *kwds);
PyObject * Record_get_obs(RinexRecord *self, PyObject *args, PyObject *kwds);


#endif //LIBRINEX_PY_RECORD_H
