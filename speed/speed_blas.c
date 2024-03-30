#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vecLib/vecLib.h>

#include "Accelerate/Accelerate.h"

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

#define D 768

float A[D * D], X[D], Y[D];

static void benchmark_sgemv(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA) {
    cblas_sgemv(order, transA, D, D, 1.0, A, D, X, 1, 0.0, Y, 1);
}

int main() {
    enum CBLAS_ORDER orders[] = {CblasRowMajor, CblasColMajor};
    enum CBLAS_TRANSPOSE transposes[] = {CblasNoTrans, CblasTrans};

    const char *orders_str[] = {"row major", "column major"};
    const char *transposes_str[] = {"not transposed", "transposed"};

    SETUP_COUNTER();

    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
            char name[128];
            sprintf(name, "SGEMV %d x %d  *  %d x 1 (%s order, A %s): %" CYCLE_TYPE "\n", D, D, D, orders_str[j],
                    transposes_str[k]);
            WRAP_FUNC(name, cycles, time0, time1, benchmark_sgemv(orders[j], transposes[k]));
        }
    }

    return 0;
}
