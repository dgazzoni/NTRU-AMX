#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "feat_dit.h"
#include "memory_alloc.h"
#ifdef KARATSUBA
#include "polymodmul_karatsuba.h"
#define MUL_FUNCTION amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs
#else
#include "polymodmul.h"
#define MUL_FUNCTION amx_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs
#endif
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

#define MAX_D 8192

int main() {
    uint16_t *xy = MEMORY_ALLOC(2 * (MAX_D / 32) * 32 * sizeof(uint16_t));
    uint16_t *x = MEMORY_ALLOC((MAX_D / 32 + 1) * 32 * sizeof(uint16_t));
    uint16_t *y = MEMORY_ALLOC((MAX_D / 32 + 1) * 32 * sizeof(uint16_t));
    unsigned char entropy_input[48] = {0};

#ifdef USE_FEAT_DIT
    set_dit_bit();
#endif

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    randombytes_init(entropy_input, NULL, 256);
    randombytes((unsigned char *)x, (MAX_D / 32) * sizeof(uint16_t));
    randombytes((unsigned char *)y, (MAX_D / 32) * sizeof(uint16_t));

    SETUP_COUNTER();

    for (int d = 192; d <= MAX_D; d += 32) {
        char name[64];
        sprintf(name, "amx_polymodmul %d x %d: %" CYCLE_TYPE "\n", d, d);
        WRAP_FUNC(name, cycles, time0, time1, MUL_FUNCTION(xy, x, y, d, (d + 31) / 32));
    }

    return 0;
}
