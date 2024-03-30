
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NTESTS 1000

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

void nist_randombytes_init(unsigned char *entropy_input, unsigned char *personalization_string, int security_strength);
int nist_randombytes(unsigned char *x, unsigned long long xlen);

void opt_randombytes_init(unsigned char *entropy_input, unsigned char *personalization_string, int security_strength);
int opt_randombytes(unsigned char *x, unsigned long long xlen);

#define BENCHMARK(func, name, size) \
    WRAP_FUNC(#func " " name ": " CYCLE_TYPE "\n", cycles, time0, time1, func(buf, size))

#define BENCHMARKS(name, size)                       \
    do {                                             \
        BENCHMARK(nist_randombytes, name, size);     \
        BENCHMARK(opt_randombytes, name, size);      \
        printf("\n");                                \
    }                                                \
    while (0)

int main() {
    unsigned char buf[(30 * 820 + 7) / 8];
    uint8_t entropy_input[48] = {0};

    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }

    nist_randombytes_init(entropy_input, NULL, 256);
    opt_randombytes_init(entropy_input, NULL, 256);

    SETUP_COUNTER();

    BENCHMARKS("hps2048509", (30 * 508 + 7) / 8);
    BENCHMARKS("hps2048677", (30 * 676 + 7) / 8);
    BENCHMARKS("hps4096821", (30 * 820 + 7) / 8);
    BENCHMARKS("hrss701", (30 * 700 + 7) / 8);

    return 0;
}
