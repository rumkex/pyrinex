#include "py_record.h"
#include "py_generator.h"
#include "util.h"
#include <structmember.h>
#include <datetime.h>

static PyMemberDef Record_members[] = {
        {"time", T_OBJECT_EX, offsetof(RinexRecord, time), 0, "Epoch time"},
        {NULL}  /* Sentinel */
};

static PyMethodDef Record_methods[] = {
        {"obs", (PyCFunction)Record_get_obs, METH_VARARGS | METH_KEYWORDS,
         "Return observation for specified sat ID and measurement type, or None if missing" },
        {NULL}  /* Sentinel */
};


PyTypeObject RinexRecordType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "rinex.Record",            /* tp_name */
        sizeof(RinexRecord),       /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)Record_dealloc,/* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_compare */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,       /* tp_flags */
        "Record object",           /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        Record_methods,            /* tp_methods */
        Record_members,            /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)Record_init,     /* tp_init */
        0,                         /* tp_alloc */
        (newfunc)Record_new,       /* tp_new */
};

void Record_dealloc(RinexRecord* self)
{
    Py_XDECREF(self->time);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject * Record_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RinexRecord *self;

    self = (RinexRecord *)type->tp_alloc(type, 0);

    self->time = Py_None;
    Py_INCREF(Py_None);

    self->header = Py_None;
    Py_INCREF(Py_None);

    return (PyObject*)self;
}

int Record_init(RinexRecord *self, PyObject *args, PyObject *kwds)
{
    RinexGenerator *generator = NULL;
    char* kwdFormat[] = { "generator", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwdFormat, &RinexGeneratorType, &generator))
        return -1;

    char buffer[BUFFER_SIZE];

    if (!Generator_read_line((PyObject *) generator, buffer, sizeof(buffer))) {
        PyErr_SetString(PyExc_StopIteration, "End of file");
        return -1;
    }

    if (buffer[0] != '>') {
        PyErr_SetString(PyExc_Exception, "Record does not start with epoch line");
        return -1;
    }

    char* end;
    int year = (int)strntol(&buffer[2], &end, 4);
    int month = (int)strntol(&buffer[7], &end, 2);
    int day = (int)strntol(&buffer[10], &end, 2);
    int hour = (int)strntol(&buffer[13], &end, 2);
    int min = (int)strntol(&buffer[16], &end, 2);
    int sec = (int)strntol(&buffer[19], &end, 2);
    int usec = (int)strntol(&buffer[22], &end, 7)/10;
    self->time = PyDateTime_FromDateAndTime(year, month, day, hour, min, sec, usec);

    int nsats = (int)strntol(&buffer[32], &end, 3);
    if (end - &buffer[32] < 3)
    {
        PyErr_SetString(PyExc_Exception, "Could not read epoch header");
        return -1;
    }

    for (int i = 0; i < nsats; i++) {
        if (Generator_is_eof((PyObject *) generator)) {
            PyErr_SetString(PyExc_Exception, "Unexpected EOF");
            return -1;
        }

        if (!Generator_read_line((PyObject *) generator, buffer, sizeof(buffer))) {
            PyErr_SetString(PyExc_Exception, "Unexpected EOF");
            return -1;
        }
        char* sysIdx = strchr(sysList, buffer[0]);
        if (sysIdx == NULL) {
            PyErr_Format(PyExc_Exception, "Unknown system %c at line %d", buffer[0], generator->line_number);
            return -1;
        }

        rinex_row* row = &self->rows[i];

        row->satSys = (int)(sysIdx - sysList);
        row->satId = (int)strntol(&buffer[1], NULL, 2);

        if (!Generator_is_sat_requested((PyObject*)generator, row->satSys, row->satId))
            continue;

        PyObject* typeList = PyDict_GetItem(generator->obsTypes, PyUnicode_FromStringAndSize(sysIdx, 1));

        if (typeList == NULL) {
            PyErr_Format(PyExc_Exception, "System %c has no record types registered", *sysIdx);
            return -1;
        }

        ssize_t typeCount = PyList_Size(typeList);
        for (ssize_t obsId = 0; obsId < typeCount; obsId++)
        {
            row->values[obsId].present = -1;

            char* obsType = PyUnicode_AsUTF8(PyList_GET_ITEM(typeList, obsId));

            if (!obsType) return -1;

            if (!Generator_is_obs_requested((PyObject*)generator, row->satSys, obsType))
                continue;

            ssize_t offset = 3 + obsId * 16;
            if (buffer[offset] == '\0')
                break;
            row->values[obsId].present = 1;
            row->values[obsId].value = strntod(&buffer[offset], &end, 14);
            if (end == &buffer[offset]) {
                row->values[obsId].present = -1;
                continue;
            }
            if (end - &buffer[offset] < 14) {
                PyErr_Format(PyExc_Exception, "Value truncated at position %d at line %d", offset, generator->line_number);
                return -1;
            }

            row->values[obsId].lli = -1;
            row->values[obsId].ssi = -1;
            if (buffer[offset+14] == '\0')
                break;
            if (buffer[offset+14] != ' ')
                row->values[obsId].lli = (short)(buffer[offset+14] - '0');
            if (buffer[offset+15] == '\0')
                break;
            if (buffer[offset+15] != ' ')
                row->values[obsId].ssi = (short)(buffer[offset+15] - '0');
        }
        row->values[typeCount].present = 0;
        self->rows[nsats].satId = 0;
    }
    return 0;
}

PyObject * Record_get_obs(RinexRecord* self, PyObject *args, PyObject *kwds)
{
    char* satSys = NULL;
    int satId = -1;
    int obsId = -1;

    if (!PyArg_ParseTuple(args, "sii", &satSys, &satId, &obsId)) {
        return NULL;
    }

    char* foundSys = strchr(sysList, satSys[0]);
    if (!foundSys)
    {
        PyErr_Format(PyExc_ValueError, "System %c is unknown", satSys[0]);
        return NULL;
    }

    size_t sysId = foundSys - sysList;

    for (int r = 0; r < MAX_SATS; r++)
    {
        rinex_row* row = &self->rows[r];
        if (row->satId == 0)
            break;

        if (row->satId == satId && row->satSys == sysId) {
            if (row->values[obsId].present == 1)
                return PyTuple_Pack(3,
                                    PyFloat_FromDouble(row->values[obsId].value/1000.),
                                    PyLong_FromLong(row->values[obsId].lli),
                                    PyLong_FromLong(row->values[obsId].ssi));
            else {
                Py_RETURN_NONE;
            }
        }
    }
    Py_RETURN_NONE;
}

PyObject *init_record_type(PyObject *m) {
    PyDateTime_IMPORT;

    if (PyType_Ready(&RinexRecordType) < 0)
        return NULL;

    Py_INCREF(&RinexRecordType);
    PyModule_AddObject(m, "Record", (PyObject*)&RinexRecordType);
    return m;
}
