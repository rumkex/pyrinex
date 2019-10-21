#include "py_sp3.h"
#include "util.h"
#include <structmember.h>
#include <datetime.h>

static PyMemberDef Ephemeris_members[] = {
        {"start", T_OBJECT_EX, offsetof(Ephemeris, start_time), 0, "Start time"},
        {"sats", T_OBJECT_EX, offsetof(Ephemeris, sats), 0, "Satellites available"},
        {"epoches", T_OBJECT_EX, offsetof(Ephemeris, epoches), 0, "Epoches available"},
        {"records", T_OBJECT_EX, offsetof(Ephemeris, records), 0, "Data records"},
        {NULL}  /* Sentinel */
};

static PyMethodDef Ephemeris_methods[] = {
//        {"position_at", (PyCFunction)Ephemeris_get_position_at, METH_VARARGS,
//                "Return satellite position at desired time, or None if not applicable" },
        {NULL}  /* Sentinel */
};

PyTypeObject EphemerisType = {
        { PyObject_HEAD_INIT(NULL) },
        "rinex.Ephemeris",              /* tp_name */
        sizeof(Ephemeris),              /* tp_basicsize */
        0,                              /* tp_itemsize */
        (destructor)Ephemeris_dealloc,  /* tp_dealloc */
        0,                              /* tp_print */
        0,                              /* tp_getattr */
        0,                              /* tp_setattr */
        0,                              /* tp_compare */
        0,                              /* tp_repr */
        0,                              /* tp_as_number */
        0,                              /* tp_as_sequence */
        0,                              /* tp_as_mapping */
        0,                              /* tp_hash */
        0,                              /* tp_call */
        0,                              /* tp_str */
        0,                              /* tp_getattro */
        0,                              /* tp_setattro */
        0,                              /* tp_as_buffer*/
        Py_TPFLAGS_DEFAULT,             /* tp_flags */
        "SP3 ephemeris storage.",       /* tp_doc */
        0,                              /* tp_traverse */
        0,                              /* tp_clear */
        0,                              /* tp_richcompare */
        0,                              /* tp_weaklistoffset */
        0,                              /* tp_iter */
        0,                              /* tp_iternext */
        Ephemeris_methods,              /* tp_methods */
        Ephemeris_members,              /* tp_members */
        0,                              /* tp_getset */
        0,                              /* tp_base */
        0,                              /* tp_dict */
        0,                              /* tp_descr_get */
        0,                              /* tp_descr_set */
        0,                              /* tp_dictoffset */
        (initproc)Ephemeris_init,       /* tp_init */
        PyType_GenericAlloc,            /* tp_alloc */
        (newfunc)Ephemeris_new          /* tp_new */
};

PyObject * Ephemeris_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Ephemeris *self;

    self = (Ephemeris*)type->tp_alloc(type, 0);

    self->records = Py_None;
    Py_INCREF(Py_None);

    self->epoches = Py_None;
    Py_INCREF(Py_None);

    self->sats = Py_None;
    Py_INCREF(Py_None);

    return (PyObject*)self;
}

PyObject *Ephemeris_load_tables(Ephemeris *self, char* fn);

int Ephemeris_init(Ephemeris *self, PyObject *args) {
    PyObject *seq = PySequence_Fast(args, "expected a sequence of filenames");
    size_t len = PySequence_Size(args);

    for (size_t i = 0; i < len; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        char *filename = PyUnicode_AsUTF8(item);

        if (!filename)
            return -1;

        PyObject* frames = Ephemeris_load_tables(self, filename);
        if (!frames)
            return -1;
    }
    Py_DECREF(seq);
    return 0;
}

void skip_lines(FILE* file, int nlines, char* magic) {
    char buffer[128];
    for (int line = 0; line < nlines; line++) {
        fgets(buffer, sizeof(buffer), file);
        assert(strncmp(buffer, magic, 2) == 0);
    }
}

PyObject *read_sp3_datetime(char* buffer) {
    int year = (int)strntol(&buffer[3], NULL, 4);
    int month = (int)strntol(&buffer[8], NULL, 2);
    int day = (int)strntol(&buffer[11], NULL, 2);
    int hour = (int)strntol(&buffer[14], NULL, 2);
    int min = (int)strntol(&buffer[17], NULL, 2);
    int sec = (int)strntol(&buffer[20], NULL, 2);
    int us = (int)strntol(&buffer[23], NULL, 6);

    return PyDateTime_FromDateAndTime(year, month, day, hour, min, sec, us);
}

PyObject *Ephemeris_load_tables(Ephemeris *self, char* fn) {
    FILE* file = fopen(fn, "rb");
    char buffer[80];

    // Line 1: has a lot of important data
    fgets(buffer, sizeof(buffer), file);

    // Check if it's an actual SP3 file we're reading
    if (strncmp(buffer, "#cP", 3) != 0)
    {
        PyErr_Format(PyExc_Exception, "Expected SP3C Position file header at line 1");
        return NULL;
    }

    // First epoch date
    self->start_time = read_sp3_datetime(buffer);
    if (!self->start_time) {
        return NULL;
    }

    // The number of epoches
    ssize_t nrec = (ssize_t)strntol(&buffer[32], NULL, 7);
    assert(nrec >= 0);

    // Line 2: skipping it
    fgets(buffer, sizeof(buffer), file);
    assert(strncmp(buffer, "##", 2) == 0);

    // Lines 3-7: get satellite IDs

    PyArray_Descr *s3_descr;
    PyArray_DescrConverter(PyUnicode_FromString("S3"), &s3_descr);

    long nsats = 0;
    int n = 0;
    for (int line = 3; line <= 7; line++) {
        fgets(buffer, sizeof(buffer), file);
        assert(strncmp(buffer, "+ ", 2) == 0);
        if (line == 3) {
            nsats = strntol(&buffer[4], NULL, 2);
            assert(nsats >= 0);
            assert(nsats <= SP3_MAX_SATS);

            Py_XDECREF(self->sats);
            self->sats = PyArray_SimpleNewFromDescr(1, &nsats, s3_descr);
            if (!self->sats) return NULL;
        }
        for (int col = 0; col < 17; col++) {
            if (n < nsats) {
                char* ptr = PyArray_GETPTR1(self->sats, n);
                strncpy(ptr, &buffer[9 + col*3], 3);
            }
            n++;
        }
    }
    // Lines 8-12: accuracy, skipping
    skip_lines(file, 5, "++");

    // Lines 13-18: extra fields, skipping
    skip_lines(file, 2, "%c");
    skip_lines(file, 2, "%f");
    skip_lines(file, 2, "%i");

    // Lines 19-22: comments, skipping
    skip_lines(file, 4, "/*");

    long dims[3] = {nrec, nsats, 3};
    Py_XDECREF(self->records);
    self->records = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    if (!self->records) return NULL;

    Py_XDECREF(self->epoches);

    PyArray_Descr *dt_descr;
    PyArray_DescrConverter(PyUnicode_FromString("datetime64[s]"), &dt_descr);

    self->epoches = PyArray_SimpleNewFromDescr(1, &nrec, dt_descr);
    if (!self->epoches) return NULL;

    // Start reading epoches
    for (ssize_t i = 0; i < nrec; i++) {
        // Read epoch header
        fgets(buffer, sizeof(buffer), file);
        assert(strncmp(buffer, "* ", 2) == 0);
        PyObject *epochdate = read_sp3_datetime(buffer);
        if (!epochdate)
            return NULL;
        if (PyArray_SETITEM(self->epoches, PyArray_GETPTR1(self->epoches, i), epochdate))
            return NULL;

        // Read satellite positions
        for (int j = 0; j < nsats; j++) {
            fgets(buffer, sizeof(buffer), file);
            assert(buffer[0] == 'P');
            char* satptr = PyArray_GETPTR1(self->sats, j);
            assert(strncmp(&buffer[1], satptr, 3) == 0);
            for (int k = 0; k < 3; k++)
            {
                double coord = strntod(&buffer[4 + 14*k], NULL, 14) * 1000;
                double* ptr = PyArray_GETPTR3(self->records, i, j, k);
                *ptr = coord;
            }
        }
    }

    // Read EOF
    fgets(buffer, sizeof(buffer), file);
    assert(strncmp(buffer, "EOF", 3) == 0);

    fclose(file);

    return self->records;
}


void Ephemeris_dealloc(Ephemeris* self)
{
    Py_XDECREF(self->start_time);
    Py_XDECREF(self->records);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


PyObject *init_ephemeris_type(PyObject *m) {
    PyDateTime_IMPORT;

    if (PyType_Ready(&EphemerisType) < 0)
        return NULL;

    Py_INCREF(&EphemerisType);
    PyModule_AddObject(m, "Ephemeris", (PyObject*)&EphemerisType);

    import_array();
    return m;
}