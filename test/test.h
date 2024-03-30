#ifndef TEST_H
#define TEST_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "gtest/gtest.h"

// https://stackoverflow.com/a/10062016/523079
template <typename T, size_t size>
::testing::AssertionResult ArraysMatch(const T (&expected)[size], const T (&actual)[size]) {
    for (size_t i(0); i < size; ++i) {
        if (expected[i] != actual[i]) {
            return ::testing::AssertionFailure()
                   << "expected[" << i << "] (" << +expected[i] << ") != actual[" << i << "] (" << +actual[i] << ")";
        }
    }

    return ::testing::AssertionSuccess();
}

template <class T1, class T2>
::testing::AssertionResult ArraysMatch(const T1 *expected, const T2 *actual, size_t array_len) {
    for (size_t i(0); i < array_len; ++i) {
        if (expected[i] != actual[i]) {
            return ::testing::AssertionFailure()
                   << "expected[" << i << "] (" << +expected[i] << ") != actual[" << i << "] (" << +actual[i] << ")";
            ;
        }
    }

    return ::testing::AssertionSuccess();
}

static inline void gen_random(uint16_t v[], size_t length) {
    for (size_t i = 0; i < length; i++) {
        v[i] = rand() & 0xFFFF;
    }
}

static inline void ref_poly_mul_mod_65536_u16_nxn_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[], size_t n) {
    memset(xy, 0, 2 * n * sizeof(uint16_t));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            xy[i + j] += x[i] * y[j];
        }
    }
}

static inline void ref_poly_mul_mod_65536_mod_x_d_minus_1_u16_nxn_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[],
                                                                         size_t d) {
    memset(xy, 0, d * sizeof(uint16_t));

    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++) {
            xy[(i + j) % d] += x[i] * y[j];
        }
    }
}

#endif  // TEST_H
