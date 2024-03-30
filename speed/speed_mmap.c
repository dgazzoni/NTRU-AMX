#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "api.h"
#include "feat_dit.h"
#include "params.h"
#include "owcpa.h"
#include "rng.h"

#ifndef NTESTS
#define NTESTS 1024
#endif

uint64_t time0, time1;
uint64_t cycles[NTESTS];

#ifdef __APPLE__

#include "m1cycles.h"
#define SETUP_COUNTER() {(void)cycles; setup_rdtsc();}
#define CYCLE_TYPE "%lld"
#define GET_TIME rdtsc()

#else

#include "hal.h"
#define SETUP_COUNTER() {}
#define CYCLE_TYPE "%ld"
#define GET_TIME hal_get_time()

#endif

#undef __MEDIAN__
#define __AVERAGE__

#ifdef __AVERAGE__

#define LOOP_INIT(__clock0, __clock1) \
    {                                 \
        __clock0 = GET_TIME;          \
    }
#define LOOP_TAIL(__f_string, records, __clock0, __clock1)  \
    {                                                       \
        __clock1 = GET_TIME;                                \
        printf(__f_string, (__clock1 - __clock0) / NTESTS); \
    }
#define BODY_INIT(__clock0, __clock1) \
    {}
#define BODY_TAIL(records, __clock0, __clock1) \
    {}

#elif defined(__MEDIAN__)

static int cmp_uint64(const void *a, const void *b) {
    return ((*((const uint64_t *)a)) - ((*((const uint64_t *)b))));
}

#define LOOP_INIT(__clock0, __clock1) \
    {}
#define LOOP_TAIL(__f_string, records, __clock0, __clock1)    \
    {                                                         \
        qsort(records, NTESTS, sizeof(uint64_t), cmp_uint64); \
        printf(__f_string, records[NTESTS >> 1]);             \
    }
#define BODY_INIT(__clock0, __clock1) \
    {                                 \
        __clock0 = GET_TIME;          \
    }
#define BODY_TAIL(records, __clock0, __clock1) \
    {                                          \
        __clock1 = GET_TIME;                   \
        records[i] = __clock1 - __clock0;      \
    }

#endif

#define WRAP_FUNC(__f_string, records, __clock0, __clock1, func) \
    {                                                            \
        /* warmup */                                             \
        func;                                                    \
        LOOP_INIT(__clock0, __clock1);                           \
        for (size_t i = 0; i < NTESTS; i++) {                    \
            BODY_INIT(__clock0, __clock1);                       \
            func;                                                \
            BODY_TAIL(records, __clock0, __clock1);              \
        }                                                        \
        LOOP_TAIL(__f_string, records, __clock0, __clock1);      \
    }
int main()
{
    unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0};
    unsigned char sk[CRYPTO_SECRETKEYBYTES] = {0};
    unsigned char ct[CRYPTO_CIPHERTEXTBYTES] = {0};
    unsigned char key_a[CRYPTO_BYTES] = {0};
    unsigned char key_b[CRYPTO_BYTES] = {0};
    uint8_t seed[NTRU_SAMPLE_FG_BYTES];
    uint8_t rm[NTRU_OWCPA_MSGBYTES];
    poly r, m;
    unsigned char entropy_input[48] = {0};

#ifdef USE_FEAT_DIT
    set_dit_bit();
#endif

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    randombytes_init(entropy_input, NULL, 256);
    randombytes(seed, NTRU_SAMPLE_FG_BYTES);

    SETUP_COUNTER();

    WRAP_FUNC("crypto_kem_keypair: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            crypto_kem_keypair(pk, sk));
    WRAP_FUNC("crypto_kem_enc: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            crypto_kem_enc(ct, key_b, pk));
    WRAP_FUNC("crypto_kem_dec: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            crypto_kem_dec(key_a, ct, sk));

    WRAP_FUNC("owcpa_keypair: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            owcpa_keypair(pk, sk, seed));
    WRAP_FUNC("owcpa_enc: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            owcpa_enc(ct, &r, &m, pk));
    WRAP_FUNC("owcpa_dec: " CYCLE_TYPE "\n",
            cycles, time0, time1,
            owcpa_dec(rm, ct, sk));

  return 0;
}
