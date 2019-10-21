#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include "util.h"

long strntol(char *ptr, char **end, int nchars) {
    int read = 0, valread = 0;
    long result = 0;
    int sign = 1;
    while (
        (read < nchars) &&
        (((*ptr == ' ' || *ptr == '-') && valread == 0) ||
         (*ptr >= '0' && *ptr <= '9'))
    )
    {
        if (*ptr != ' ')
        {
            if (*ptr == '-') {
                sign = -1;
            }
            else {
                result = 10*result + (*ptr - '0');
            }
            valread++;
        }
        ptr++; read++;
    }
    if (end != NULL) {
        if (valread > 0) {
            *end = ptr;
        } else {
            *end = ptr - read;
        }
    }
    return sign*result;
}

long strntod(char *ptr, char **end, int nchars) {
    char *real_end = NULL;
    long integer = strntol(ptr, &real_end, nchars);
    if (*real_end != '.') {
        *end = real_end;
        return integer;
    }
    char *dot = real_end+1;

    int fraclen = (int)(nchars-(real_end-ptr))-1;

    long frac = strntol(dot, &real_end, fraclen);
    long readfrac = real_end - dot;

    if (integer < 0) frac = -frac;

    while (readfrac > 0) {
        readfrac--;
        integer *= 10;
    }

    if (end != NULL) *end = real_end;

    return integer + frac;
}

int strmincmp(const char* ptr1, const char *ptr2) {
    while (*ptr1 == *ptr2 && *ptr1 != '\0' && *ptr2 != '\0')
    {
        ptr1++; ptr2++;
    }
    if (*ptr1 == '\0' || *ptr2 == '\0')
        return 0;
    return -1;
}