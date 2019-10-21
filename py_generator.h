#ifndef LIBRINEX_GENERATOR_H
#define LIBRINEX_GENERATOR_H

#include <stdio.h>
#include <Python.h>

extern const char hdrVersion[];
extern const char hdrEndOfHeader[];
extern const char hdrSysObsTypes[];
extern const char hdrApproxPosition[];
extern const char hdrComment[];

extern const char sysList[];

typedef struct {
    PyObject_HEAD;
    FILE *fp;
    int line_number;
    PyObject *version;
    PyObject *position;
    PyObject *obsTypes;
    PyObject *obsFilter;
    PyObject *satFilter;
} RinexGenerator;

extern PyTypeObject RinexGeneratorType;

PyObject * Generator_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int Generator_init(PyObject *self, PyObject *args);
PyObject* Generator_next(PyObject *self);
void Generator_dealloc(PyObject *self);

int Generator_is_sat_requested(PyObject* self, int satSys, int satId);
int Generator_is_obs_requested(PyObject* self, int satSys, const char* obsType);
char* Generator_read_line(PyObject *self, char *buffer, int buf_size);
int Generator_is_eof(PyObject *self);

PyObject *init_generator_type(PyObject *m);

#endif //LIBRINEX_GENERATOR_H
