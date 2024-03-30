#ifndef crypto_sort_H
#define crypto_sort_H

#include <stdint.h>
#include <stddef.h>

// Modified in NTRU-sampling to match PQC_NEON
#ifndef CRYPTO_NAMESPACE
#define CRYPTO_NAMESPACE(s) s
#endif

#define crypto_sort_int32 CRYPTO_NAMESPACE(crypto_sort_int32)

void crypto_sort_int32(int32_t *, size_t);
void crypto_sort_uint32(uint32_t *, size_t);

#endif
