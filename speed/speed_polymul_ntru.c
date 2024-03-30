#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "feat_dit.h"
#include "memory_alloc.h"
#include "polymul.h"
#include "rng.h"

#define NTESTS 1000

uint64_t time0, time1;
uint64_t cycles[NTESTS];

#ifdef __APPLE__

#include "m1cycles.h"
#define SETUP_COUNTER() \
    {                   \
        (void)cycles;   \
        setup_rdtsc();  \
    }
#define CYCLE_TYPE "%lld"
#define GET_TIME rdtsc()

#else

#include "hal.h"
#define SETUP_COUNTER() \
    {}
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

int main() {
    uint16_t *xy = MEMORY_ALLOC(2 * 26 * 32 * sizeof(uint16_t));
    uint16_t *x = MEMORY_ALLOC(26 * 32 * sizeof(uint16_t));
    uint16_t *y = MEMORY_ALLOC(26 * 32 * sizeof(uint16_t));

    unsigned char entropy_input[48] = {0};

#ifdef USE_FEAT_DIT
    set_dit_bit();
#endif

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    randombytes_init(entropy_input, NULL, 256);
    randombytes((unsigned char *)x, 26 * sizeof(uint16_t));
    randombytes((unsigned char *)y, 26 * sizeof(uint16_t));

    SETUP_COUNTER();

    WRAP_FUNC("amx_polymodmul 509 x 509: " CYCLE_TYPE "\n", cycles, time0, time1,
              amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, (509 + 31)/32));
    WRAP_FUNC("amx_polymodmul 677 x 677: " CYCLE_TYPE "\n", cycles, time0, time1,
              amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, (677 + 31)/32));
    WRAP_FUNC("amx_polymodmul 821 x 821: " CYCLE_TYPE "\n", cycles, time0, time1,
              amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, (821 + 31)/32));
    WRAP_FUNC("amx_polymodmul 701 x 701: " CYCLE_TYPE "\n", cycles, time0, time1,
              amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, (701 + 31)/32));

    return 0;
}
