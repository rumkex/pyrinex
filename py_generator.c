#include "py_generator.h"
#include "py_record.h"
#include "util.h"
#include <structmember.h>

const char hdrVersion[]        = "RINEX VERSION / TYPE";
const char hdrEndOfHeader[]    = "END OF HEADER";
const char hdrSysObsTypes[]    = "SYS / # / OBS TYPES";
const char hdrApproxPosition[] = "APPROX POSITION XYZ";
const char hdrComment[]        = "COMMENT";

const char sysList[] = "GREJCIS";


static PyMemberDef Generator_members[] = {
        {"line_number", T_INT, offsetof(RinexGenerator, line_number), 0, "Current line number"},
        {"version", T_OBJECT_EX, offsetof(RinexGenerator, version), 0, "RINEX version"},
        {"position",   T_OBJECT_EX, offsetof(RinexGenerator, position), 0, "Receiver position"},
        {"obs_types",  T_OBJECT_EX, offsetof(RinexGenerator, obsTypes), 0, "Observation type dictionary"},
        {"obs_filter", T_OBJECT_EX, offsetof(RinexGenerator, obsFilter), 0, "Selected observation types"},
        {"sat_filter", T_OBJECT_EX, offsetof(RinexGenerator, satFilter), 0, "Selected satellites"},
        {NULL}  /* Sentinel */
};

PyTypeObject RinexGeneratorType = {
        { PyObject_HEAD_INIT(NULL) },
        "rinex.Generator",              /*tp_name*/
        sizeof(RinexGenerator),         /*tp_basicsize*/
        0,                              /*tp_itemsize*/
        (destructor)Generator_dealloc,  /*tp_dealloc*/
        0,                              /*tp_print*/
        0,                              /*tp_getattr*/
        0,                              /*tp_setattr*/
        0,                              /*tp_compare*/
        0,                              /*tp_repr*/
        0,                              /*tp_as_number*/
        0,                              /*tp_as_sequence*/
        0,                              /*tp_as_mapping*/
        0,                              /*tp_hash */
        0,                              /*tp_call*/
        0,                              /*tp_str*/
        0,                              /*tp_getattro*/
        0,                              /*tp_setattro*/
        0,                              /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT,             /* tp_flags */
        "Internal iterator object.",    /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        PyObject_SelfIter,              /* tp_iter */
        (iternextfunc)Generator_next,   /* tp_iternext */
        0,                              /* tp_methods */
        Generator_members,              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc)Generator_init,       /* tp_init */
        PyType_GenericAlloc,            /* tp_alloc */
        (newfunc)Generator_new          /* tp_new */
};

int Generator_init(PyObject *self, PyObject *args) {
    char* path = NULL;
    PyObject *meas = NULL, *svs = NULL;
    if (!PyArg_ParseTuple(args, "sOO", &path, &meas, &svs))
        return -1;

    RinexGenerator *p = (RinexGenerator*)self;

    p->line_number = 0;
    p->fp = fopen(path, "rb");

    if (!p->fp) {
        PyErr_Format(PyExc_FileNotFoundError, "Could not open file %s", path);
        return -1;
    }

    int is_rinex = 0;

    while (1)
    {
        char buffer[BUFFER_SIZE];
        Generator_read_line(self, buffer, sizeof(buffer));
        if (p->line_number > 1 && !is_rinex)
        {
            PyErr_Format(PyExc_Exception, "Expected RINEX version header at line 1");
            return -1;
        }

        if (Generator_is_eof(self)) {
            PyErr_Format(PyExc_Exception, "Unexpected end of file at line %d", p->line_number);
            return -1;
        }

        if (strlen(buffer) < 60) {
            PyErr_Format(PyExc_Exception, "Line too short (<60 symbols) at line %d", p->line_number);
            return -1;
        }

        if (strmincmp(&buffer[60], hdrVersion) == 0) {
            char *end;
            long version = strntol(buffer, &end, 10);
            if (*end != '.')
            {
                PyErr_Format(PyExc_Exception, "Malformed RINEX version at line %d", p->line_number);
                return -1;
            }
            long versionMinor = strntol(end+1, &end, 2);
            p->version = PyTuple_Pack(2,
                    PyLong_FromLong(version),
                    PyLong_FromLong(versionMinor)
            );
            is_rinex = 1;
        }
        else if (strmincmp(&buffer[60], hdrSysObsTypes) == 0) {
            char sys = buffer[0];
            char* foundSys = strchr(sysList, sys);
            if (foundSys == NULL) {
                // Unknown system
                PyErr_Format(PyExc_Exception, "Unknown system %c at line %d", sys, p->line_number);
                return -1;
            }

            char* end;
            int typesTotal = (int)strntol(&buffer[1], &end, 5);

            if (typesTotal > MAX_OBS_TYPES) {
                PyErr_Format(PyExc_Exception, "Too many observation types for system %c at line %d",
                             sys, p->line_number);
                return -1;
            }

            PyObject* typesList = PyList_New(typesTotal);

            for (int i = 0; i < typesTotal; i++) {
                int idx = i % 13; // 13 obs types per line

                if (idx == 0 && i != 0) {
                    Generator_read_line(self, buffer, sizeof(buffer));
                    if (strmincmp(&buffer[60], hdrSysObsTypes) != 0) {
                        PyErr_Format(PyExc_Exception, "Expected %s header continuation but got %s at line %d",
                                     hdrSysObsTypes, &buffer[60], p->line_number);
                        return -1;
                    }
                }

                PyList_SET_ITEM(typesList, i, PyUnicode_FromStringAndSize(&buffer[7 + 4*idx], 3));
            }
            PyDict_SetItem(p->obsTypes, PyUnicode_FromStringAndSize(&sys, 1), typesList);
            Py_DECREF(typesList);
        }
        else if (strmincmp(&buffer[60], hdrApproxPosition) == 0)
        {
            Py_XDECREF(p->position);
            p->position = PyTuple_Pack(3,
                                          PyFloat_FromDouble(strntod(&buffer[0], NULL, 14)/10000.),
                                          PyFloat_FromDouble(strntod(&buffer[14], NULL, 14)/10000.),
                                          PyFloat_FromDouble(strntod(&buffer[28], NULL, 14)/10000.)
            );
            if (!p->position) {
                return -1;
            }
        }
        else if (strmincmp(&buffer[60], hdrEndOfHeader) == 0)
            break;
    }

    if (meas && meas != Py_None) {
        if (!PySequence_Check(meas)) {
            return -1;
        }
        Py_XDECREF(p->obsFilter);
        p->obsFilter = PySet_New(meas);
        if (!p->obsFilter) return -1;
    }

    if (svs && svs != Py_None) {
        if (!PySequence_Check(svs)) {
            return -1;
        }
        Py_XDECREF(p->satFilter);
        p->satFilter = PySet_New(svs);
        if (!p->satFilter) return -1;
    }

    return 0;
}

PyObject* Generator_next(PyObject *self)
{
    RinexGenerator *p = (RinexGenerator *)self;

    if (feof(p->fp)) {
        fclose(p->fp);
        p->fp = NULL;
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    PyObject *argList = Py_BuildValue("(O)", self);
    RinexRecord *obj = (RinexRecord*)PyObject_CallObject((PyObject *) &RinexRecordType, argList);

    if (!obj) {
        return NULL;
    }

    Py_DECREF(argList);

    return (PyObject*)obj;
}

PyObject * Generator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RinexGenerator *self = (RinexGenerator *)type->tp_alloc(type, 0);

    self->fp = NULL;
    self->line_number = 0;

    self->version = Py_None;
    Py_INCREF(Py_None);

    self->position = Py_None;
    Py_INCREF(Py_None);

    self->obsTypes = PyDict_New();
    if (!self->obsTypes) return NULL;

    self->obsFilter = PySet_New(NULL);
    if (!self->obsFilter) return NULL;

    self->satFilter = PySet_New(NULL);
    if (!self->satFilter) return NULL;

    return (PyObject*)self;
}

void Generator_dealloc(PyObject *self) {
    RinexGenerator *p = (RinexGenerator *)self;
    if (p->fp != NULL) fclose(p->fp);
    Py_XDECREF(p->position);
    Py_XDECREF(p->obsTypes);
    Py_TYPE(self)->tp_free(self);
}


int Generator_is_obs_requested(PyObject* self, int satSys, const char* obsType) {
    RinexGenerator *p = (RinexGenerator*)self;

    if (p->obsFilter != NULL && PySet_Size(p->obsFilter) > 0) {
        char obsSysType[5];
        obsSysType[0] = sysList[satSys];
        memcpy(obsSysType+1, obsType, 4);

        PyObject* handle = PyUnicode_FromString(obsSysType);

        if (!PySet_Contains(p->obsFilter, handle)) {
            return 0;
        }

        Py_DECREF(handle);
    }
    return 1;
}

int Generator_is_sat_requested(PyObject* self, int satSys, int satId) {
    RinexGenerator *p = (RinexGenerator*)self;

    if (p->satFilter != NULL && PySet_Size(p->satFilter) > 0) {
        PyObject* handle = PyTuple_Pack(2, PyUnicode_FromStringAndSize(&sysList[satSys], 1), PyLong_FromLong(satId));

        if (!PySet_Contains(p->satFilter, handle)) {
            return 0;
        }

        Py_DECREF(handle);
    }
    return 1;
}

char* Generator_read_line(PyObject *self, char *buffer, int buf_size) {
    RinexGenerator *p = (RinexGenerator*)self;
    p->line_number++;
    return fgets(buffer, buf_size, p->fp);
}

int Generator_is_eof(PyObject *self) {
    RinexGenerator *p = (RinexGenerator*)self;
    return feof(p->fp);
}


PyObject *init_generator_type(PyObject *m) {
    if (PyType_Ready(&RinexGeneratorType) < 0)
        return NULL;

    Py_INCREF(&RinexGeneratorType);
    PyModule_AddObject(m, "Generator", (PyObject*)&RinexGeneratorType);
    return m;
}