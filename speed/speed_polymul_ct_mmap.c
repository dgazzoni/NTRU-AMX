

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

#ifndef ITERS
#define ITERS 1000000
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
        __clock0 = GET_TIME;          \
    }
#define LOOP_TAIL(result, records, __clock0, __clock1) \
    {                                                  \
        __clock1 = GET_TIME;                           \
        result = (__clock1 - __clock0) / NTESTS;       \
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
#define BODY_INIT(__clock0, __clock1) \
    {                                 \
        __clock0 = GET_TIME;          \
    }
#define LOOP_TAIL(result, records, __clock0, __clock1)        \
    {                                                         \
        qsort(records, NTESTS, sizeof(uint64_t), cmp_uint64); \
        result = records[NTESTS >> 1];                        \
    }
#define BODY_TAIL(records, __clock0, __clock1) \
    {                                          \
        __clock1 = GET_TIME;                   \
        records[i] = __clock1 - __clock0;      \
    }

#endif

poly *src1, *src2, *des, *rnd1, *rnd2;

__attribute__((constructor)) void alloc_polys(void) {
    src1 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    src2 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    des = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    rnd1 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    rnd2 = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
}

typedef struct {
    uint64_t order : 1; /* 0: zeros -> random; 1: random -> zeros */
    uint64_t cycles1 : 31;
    uint64_t cycles2 : 31;
} measurement_t;

int main() {
    static measurement_t c[ITERS];
    static uint8_t sel[ITERS / 8];
    static unsigned char entropy_input[48];

#if USE_FEAT_DIT
    set_dit_bit();
#endif

    SETUP_COUNTER();

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    randombytes_init(entropy_input, NULL, 256);

    randombytes((uint8_t *)rnd1, NTRU_N * sizeof(uint16_t));
    randombytes((uint8_t *)rnd2, NTRU_N * sizeof(uint16_t));

    randombytes(sel, sizeof(sel));

    for (int k = 0; k < ITERS; k++) {
        uint16_t mask = (uint16_t)(-(int16_t)((sel[k >> 3] >> (k & 7)) & 1));

        for (int j = 0; j < NTRU_N; j++) {
            src1->coeffs[j] = rnd1->coeffs[j] & mask;
            src2->coeffs[j] = rnd2->coeffs[j] & mask;
        }

        LOOP_INIT(time0, time1);
        for (size_t i = 0; i < NTESTS; i++) {
            BODY_INIT(time0, time1);
            poly_Rq_mul(des, src1, src2);
            BODY_TAIL(cycles, time0, time1);
        }
        LOOP_TAIL(c[k].cycles1, cycles, time0, time1);

        c[k].order = mask & 1;

        mask = ~mask;

        for (int j = 0; j < NTRU_N; j++) {
            src1->coeffs[j] = rnd1->coeffs[j] & mask;
            src2->coeffs[j] = rnd2->coeffs[j] & mask;
        }

        LOOP_INIT(time0, time1);
        for (size_t i = 0; i < NTESTS; i++) {
            BODY_INIT(time0, time1);
            poly_Rq_mul(des, src1, src2);
            BODY_TAIL(cycles, time0, time1);
        }
        LOOP_TAIL(c[k].cycles2, cycles, time0, time1);
    }

    for (int i = 0; i < ITERS; i++) {
        if (c[i].order) {
            printf("%d %d\n", c[i].cycles2, c[i].cycles1);
        }
        else {
            printf("%d %d\n", c[i].cycles1, c[i].cycles2);
        }
    }

    return 0;
}
