#ifndef LIBRINEX_UTIL_H
#define LIBRINEX_UTIL_H

long strntol(char* ptr, char **end, int nchars);
long strntod(char *ptr, char **end, int nchars);
int strmincmp(const char* ptr1, const char *ptr2);

#endif //LIBRINEX_UTIL_H
