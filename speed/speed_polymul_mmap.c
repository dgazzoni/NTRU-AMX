

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"
#include "feat_dit.h"
#include "memory_alloc.h"
#include "params.h"
#include "poly.h"
#include "rng.h"

#ifndef NTESTS
#define NTESTS 1024
#endif

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
    __clock0 = GET_TIME;              \
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

poly *src1, *src2, *des, *rnd1, *rnd2;

__attribute__((constructor)) void alloc_polys(void) {
    src1 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    src2 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    des = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    rnd1 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    rnd2 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
}

int main() {
    static unsigned char entropy_input[48];
    uint16_t mask;

#ifdef USE_FEAT_DIT
    set_dit_bit();
#endif

    SETUP_COUNTER();

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    randombytes_init(entropy_input, NULL, 256);

    randombytes((uint8_t *)rnd1, NTRU_N * sizeof(uint16_t));
    randombytes((uint8_t *)rnd2, NTRU_N * sizeof(uint16_t));

    asm volatile("mov %w0, #0xFFFF\n" : "=r"(mask));
    for (int i = 0; i < NTRU_N; i++) {
        src1->coeffs[i] = rnd1->coeffs[i] & mask;
        src2->coeffs[i] = rnd2->coeffs[i] & mask;
    }

    WRAP_FUNC("poly_Rq_mul: " CYCLE_TYPE " \n", cycles, time0, time1, poly_Rq_mul(des, src1, src2));

    return 0;
}
