#ifndef RANDOMBYTES_H
#define RANDOMBYTES_H

// Modified in NTRU-sampling to match the NIST definition (returns int rather than void)
int randombytes(unsigned char *x, unsigned long long xlen);

#endif
