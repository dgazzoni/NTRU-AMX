#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "aarch64.h"
#include "gtest/gtest.h"
#include "test.h"
#include "test_amx.h"

extern "C" {
#include "polymodmul_karatsuba.h"
}

class AMX_karatsuba_poly_mul_mod_x_d_minus_1_32nx32n_coeffs : public testing::TestWithParam<int> {};

static inline std::string generate_polymulmod_n_d_test_name(const testing::TestParamInfo<int> &info) {
    int d = info.param;
    int n = ((d + 31) / 32);
    return std::string("n_") + std::to_string(n) + std::string("_d_") + std::to_string(d);
}

TEST_P(AMX_karatsuba_poly_mul_mod_x_d_minus_1_32nx32n_coeffs, single) {
    uint16_t *x, *y, *xy, *xy2;
    int d = GetParam();
    int n = ((d + 31) / 32);

    posix_memalign((void **)&x, 128, 32 * (n + 1) * sizeof(uint16_t));
    posix_memalign((void **)&y, 128, 32 * (n + 1) * sizeof(uint16_t));
    posix_memalign((void **)&xy, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy2, 128, 32 * n * sizeof(uint16_t));

    srand(1337);

    gen_random(x, 32 * n);
    gen_random(y, 32 * n);
    memset(xy2, 0, 32 * n * sizeof(uint16_t));

    ref_poly_mul_mod_65536_mod_x_d_minus_1_u16_nxn_coeffs(xy2, x, y, d);

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(xy, x, y, d, n);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy, d));

    free(x);
    free(y);
    free(xy);
    free(xy2);
}

TEST_P(AMX_karatsuba_poly_mul_mod_x_d_minus_1_32nx32n_coeffs, initialize_to_zero) {
    uint16_t *x, *y, *xy, *xy2;
    int d = GetParam();
    int n = ((d + 31) / 32);

    posix_memalign((void **)&x, 128, 32 * (n + 1) * sizeof(uint16_t));
    posix_memalign((void **)&y, 128, 32 * (n + 1) * sizeof(uint16_t));
    posix_memalign((void **)&xy, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy2, 128, 32 * n * sizeof(uint16_t));

    srand(1337);

    gen_random(x, 32 * n);
    gen_random(y, 32 * n);
    memset(xy2, 0, 32 * n * sizeof(uint16_t));

    ref_poly_mul_mod_65536_mod_x_d_minus_1_u16_nxn_coeffs(xy2, x, y, d);

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(xy, x, y, d, n);
    clobber_all_AMX_regs();
    amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(xy, x, y, d, n);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy, d));

    free(x);
    free(y);
    free(xy);
    free(xy2);
}

INSTANTIATE_TEST_SUITE_P(, AMX_karatsuba_poly_mul_mod_x_d_minus_1_32nx32n_coeffs, testing::Range(161, 822),
                         generate_polymulmod_n_d_test_name);
