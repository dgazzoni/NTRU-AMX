#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "aarch64.h"
#include "gtest/gtest.h"
#include "test.h"
#include "test_amx.h"

extern "C" {
#include "polymul.h"
}

TEST(AMX_poly_mul_32x32_coeffs, single) {
    uint16_t x[32] __attribute__((aligned(128))) = {0}, y[32] __attribute__((aligned(128))) = {0};
    uint16_t xy[64] __attribute__((aligned(128))) = {0};
    uint16_t xy2[64] __attribute__((aligned(128))) = {0};

    srand(1337);

    gen_random(x, sizeof(x) / sizeof(x[0]));
    gen_random(y, sizeof(y) / sizeof(y[0]));

    ref_poly_mul_mod_65536_u16_nxn_coeffs(xy2, x, y, sizeof(x) / sizeof(x[0]));

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32x32_coeffs(xy, x, y);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy));
}

TEST(AMX_poly_mul_32x32_coeffs, initialize_to_zero) {
    uint16_t x[32] __attribute__((aligned(128))) = {0}, y[32] __attribute__((aligned(128))) = {0};
    uint16_t xy[64] __attribute__((aligned(128))) = {0};
    uint16_t xy2[64] __attribute__((aligned(128))) = {0};

    srand(1337);

    gen_random(x, sizeof(x) / sizeof(x[0]));
    gen_random(y, sizeof(y) / sizeof(y[0]));

    ref_poly_mul_mod_65536_u16_nxn_coeffs(xy2, x, y, sizeof(x) / sizeof(x[0]));

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32x32_coeffs(xy, x, y);
    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32x32_coeffs(xy, x, y);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy));
}

class AMX_poly_mul_32nx32n_coeffs : public testing::TestWithParam<int> {};

static inline std::string generate_poly_mul_32nx32n_test_name(const testing::TestParamInfo<int> &info) {
    return std::string("n_") + std::to_string(info.param);
}

TEST_P(AMX_poly_mul_32nx32n_coeffs, single) {
    uint16_t *x, *y, *xy, *xy2;
    int n = GetParam();

    posix_memalign((void **)&x, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&y, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy, 128, 64 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy2, 128, 64 * n * sizeof(uint16_t));

    srand(1337);

    gen_random(x, 32 * n);
    gen_random(y, 32 * n);

    ref_poly_mul_mod_65536_u16_nxn_coeffs(xy2, x, y, 32 * n);

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, n);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy, 64 * n));

    free(x);
    free(y);
    free(xy);
    free(xy2);
}

TEST_P(AMX_poly_mul_32nx32n_coeffs, initialize_to_zero) {
    uint16_t *x, *y, *xy, *xy2;
    int n = GetParam();

    posix_memalign((void **)&x, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&y, 128, 32 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy, 128, 64 * n * sizeof(uint16_t));
    posix_memalign((void **)&xy2, 128, 64 * n * sizeof(uint16_t));

    srand(1337);

    gen_random(x, 32 * n);
    gen_random(y, 32 * n);

    ref_poly_mul_mod_65536_u16_nxn_coeffs(xy2, x, y, 32 * n);

#ifdef INIT_AMX
    AMX_SET();
#endif

    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, n);
    clobber_all_AMX_regs();
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(xy, x, y, n);

#ifdef INIT_AMX
    AMX_CLR();
#endif

    EXPECT_TRUE(ArraysMatch(xy2, xy, 64 * n));

    free(x);
    free(y);
    free(xy);
    free(xy2);
}

INSTANTIATE_TEST_SUITE_P(, AMX_poly_mul_32nx32n_coeffs, testing::Range(1, 27), generate_poly_mul_32nx32n_test_name);
