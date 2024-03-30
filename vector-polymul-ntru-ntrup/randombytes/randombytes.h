#ifndef RANDOMBYTES_H
#define RANDOMBYTES_H

#include <stddef.h>
#include <stdint.h>

void randombytes(uint8_t *out, size_t outlen);

#if defined(NORAND) || defined(BENCH) || defined(BENCH_RAND)

#include "rng.h"

extern unsigned char keybytes[crypto_rng_KEYBYTES];
extern unsigned char outbytes[crypto_rng_OUTPUTBYTES];
extern unsigned long long pos;

#endif

#endif
