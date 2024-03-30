#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "aarch64.h"
#include "amx.h"
#include "memory_alloc.h"
#include "polymul.h"

#ifdef CPU_M1
// M1 requires 128-byte aligned address for a load pair (https://github.com/corsix/amx/blob/main/ldst.md), which is
// being used to emulate a load quad. In general that's not the case, so we use load single.
#define AMX_LDX_QUAD(PTR, REG)                                              \
    do {                                                                    \
        for (int j = 0; j < 4; j++) {                                       \
            AMX_LDX(AMX_PTR(((uint8_t *)PTR + 64 * j)) | LDX_REG(REG + j)); \
        }                                                                   \
    }                                                                       \
    while (0)

#define AMX_LDY_QUAD(PTR, REG)                                              \
    do {                                                                    \
        for (int j = 0; j < 4; j++) {                                       \
            AMX_LDY(AMX_PTR(((uint8_t *)PTR + 64 * j)) | LDY_REG(REG + j)); \
        }                                                                   \
    }                                                                       \
    while (0)
#else
#define LDX_LOAD_QUAD (LDX_LOAD_PAIR | (UINT64_C(1) << 60))
#define LDY_LOAD_QUAD (LDY_LOAD_PAIR | (UINT64_C(1) << 60))

#define AMX_LDX_QUAD(PTR, REG)                                \
    do {                                                      \
        AMX_LDX(AMX_PTR(PTR) | LDX_REG(REG) | LDX_LOAD_QUAD); \
    }                                                         \
    while (0)

#define AMX_LDY_QUAD(PTR, REG)                                \
    do {                                                      \
        AMX_LDY(AMX_PTR(PTR) | LDY_REG(REG) | LDY_LOAD_QUAD); \
    }                                                         \
    while (0)
#endif

uint16_t *t1, *t2, *zeros;

// We allocate enough memory for the largest possible input size used in polynomial multiplication scaling tests
__attribute__((constructor)) void alloc_t(void) {
    // The first half of t1 stores the product of the low parts of input polynomials, while the second half stores the
    // product of the high parts.
    t1 = MEMORY_ALLOC(16384 * sizeof(uint16_t));
    t2 = MEMORY_ALLOC(8192 * sizeof(uint16_t));
    zeros = MEMORY_ALLOC(32 * sizeof(uint16_t));
    memset(zeros, 0, 32 * sizeof(uint16_t));
}

// Note: this function may write to elements of x and y of index up to 32 * (n + 1) - 1; arrays must be allocated
// accordingly
void amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[],
                                                                         int d, int n) {
    int i, nl = (n + 1) / 2, nh = n / 2;

    if (d % 32 != 0) {
        AMX_LDX(AMX_PTR(zeros) | LDX_REG(0));
        AMX_STX(AMX_PTR(&x[d]) | STX_REG(0));
        AMX_STX(AMX_PTR(&y[d]) | STX_REG(0));
    }

    // Split at 32 * nl

    // Prepare inputs for second half-size multiplication

    // Loop unrolling here and in the other loops is helpful for the M3, which can execute some vector instructions at a
    // rate of 2/cycle. Experiments indicate the choice of 8-way unrolling is optimal.
    for (i = 0; i < 8 * (nh / 8); i += 8) {
        AMX_LDX_QUAD(&x[32 * i], 0);
        AMX_LDX_QUAD(&x[32 * (i + 4)], 4);

        AMX_LDY_QUAD(&y[32 * i], 0);
        AMX_LDY_QUAD(&y[32 * (i + 4)], 4);

        AMX_LDZ(AMX_PTR(&x[32 * (i + nl)]) | LDZ_Z_ROW(0));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 1)]) | LDZ_Z_ROW(4));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 2)]) | LDZ_Z_ROW(8));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 3)]) | LDZ_Z_ROW(12));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 4)]) | LDZ_Z_ROW(16));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 5)]) | LDZ_Z_ROW(20));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 6)]) | LDZ_Z_ROW(24));
        AMX_LDZ(AMX_PTR(&x[32 * (i + nl + 7)]) | LDZ_Z_ROW(28));

        AMX_LDZ(AMX_PTR(&y[32 * (i + nl)]) | LDZ_Z_ROW(32));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 1)]) | LDZ_Z_ROW(36));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 2)]) | LDZ_Z_ROW(40));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 3)]) | LDZ_Z_ROW(44));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 4)]) | LDZ_Z_ROW(48));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 5)]) | LDZ_Z_ROW(52));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 6)]) | LDZ_Z_ROW(56));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl + 7)]) | LDZ_Z_ROW(60));

        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(0) | MAC16_Z_ROW(0) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(1) | MAC16_Z_ROW(4) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(2) | MAC16_Z_ROW(8) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(3) | MAC16_Z_ROW(12) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(4) | MAC16_Z_ROW(16) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(5) | MAC16_Z_ROW(20) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(6) | MAC16_Z_ROW(24) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(7) | MAC16_Z_ROW(28) | MAC16_Y_SKIP);

        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(0) | MAC16_Z_ROW(32) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(1) | MAC16_Z_ROW(36) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(2) | MAC16_Z_ROW(40) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(3) | MAC16_Z_ROW(44) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(4) | MAC16_Z_ROW(48) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(5) | MAC16_Z_ROW(52) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(6) | MAC16_Z_ROW(56) | MAC16_X_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(7) | MAC16_Z_ROW(60) | MAC16_X_SKIP);

        AMX_STZ(AMX_PTR(&t1[32 * i]) | STZ_Z_ROW(0));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 1)]) | STZ_Z_ROW(4));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2)]) | STZ_Z_ROW(8));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 3)]) | STZ_Z_ROW(12));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 4)]) | STZ_Z_ROW(16));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 5)]) | STZ_Z_ROW(20));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 6)]) | STZ_Z_ROW(24));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 7)]) | STZ_Z_ROW(28));

        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl)]) | STZ_Z_ROW(32));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 1)]) | STZ_Z_ROW(36));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 2)]) | STZ_Z_ROW(40));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 3)]) | STZ_Z_ROW(44));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 4)]) | STZ_Z_ROW(48));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 5)]) | STZ_Z_ROW(52));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 6)]) | STZ_Z_ROW(56));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl + 7)]) | STZ_Z_ROW(60));
    }

    for (; i < nh; i++) {
        AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(0));
        AMX_LDY(AMX_PTR(&y[32 * i]) | LDY_REG(0));

        AMX_LDZ(AMX_PTR(&x[32 * (i + nl)]) | LDZ_Z_ROW(0));
        AMX_LDZ(AMX_PTR(&y[32 * (i + nl)]) | LDZ_Z_ROW(32));

        AMX_MAC16(MAC16_VECTOR | MAC16_X_REG(0) | MAC16_Z_ROW(0) | MAC16_Y_SKIP);
        AMX_MAC16(MAC16_VECTOR | MAC16_Y_REG(0) | MAC16_Z_ROW(32) | MAC16_X_SKIP);

        AMX_STZ(AMX_PTR(&t1[32 * i]) | STZ_Z_ROW(0));
        AMX_STZ(AMX_PTR(&t1[32 * (i + 2 * nl)]) | STZ_Z_ROW(32));
    }

    if (nl > nh) {
        AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(0));
        AMX_LDY(AMX_PTR(&y[32 * i]) | LDY_REG(0));

        AMX_STX(AMX_PTR(&t1[32 * i]) | STX_REG(0));
        AMX_STY(AMX_PTR(&t1[32 * (i + 2 * nl)]) | STY_REG(0));
    }

    // Second half-size multiplication (middle block)
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(t2, t1, &t1[32 * 2 * nl], nl);

    // First half-size multiplication (low block)
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(t1, x, y, nl);

    // Third half-size multiplication (high block)
    amx_poly_mul_mod_65536_u16_32nx32n_coeffs(&t1[32 * 2 * nl], &x[32 * nl], &y[32 * nl], nh);

    // Subtract other two blocks to get the final value of middle block

    for (i = 0; i < 8 * ((2 * nh) / 8); i += 8) {
        AMX_LDZ(AMX_PTR(&t2[32 * i]) | LDZ_Z_ROW(0));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 1)]) | LDZ_Z_ROW(8));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 2)]) | LDZ_Z_ROW(16));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 3)]) | LDZ_Z_ROW(24));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 4)]) | LDZ_Z_ROW(32));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 5)]) | LDZ_Z_ROW(40));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 6)]) | LDZ_Z_ROW(48));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 7)]) | LDZ_Z_ROW(56));

        AMX_LDX_QUAD(&t1[32 * i], 0);
        AMX_LDX_QUAD(&t1[32 * (i + 4)], 4);

        AMX_LDY_QUAD(&t1[32 * (i + 2 * nl)], 0);
        AMX_LDY_QUAD(&t1[32 * (i + 2 * nl + 4)], 4);

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(1) | VECINT_Y_REG(1) | VECINT_Z_ROW(8) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(2) | VECINT_Y_REG(2) | VECINT_Z_ROW(16) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(3) | VECINT_Y_REG(3) | VECINT_Z_ROW(24) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(4) | VECINT_Y_REG(4) | VECINT_Z_ROW(32) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(5) | VECINT_Y_REG(5) | VECINT_Z_ROW(40) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(6) | VECINT_Y_REG(6) | VECINT_Z_ROW(48) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);
        AMX_VECINT(VECINT_X_REG(7) | VECINT_Y_REG(7) | VECINT_Z_ROW(56) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);

        AMX_STZ(AMX_PTR(&t2[32 * i]) | STZ_Z_ROW(0));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 1)]) | STZ_Z_ROW(8));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 2)]) | STZ_Z_ROW(16));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 3)]) | STZ_Z_ROW(24));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 4)]) | STZ_Z_ROW(32));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 5)]) | STZ_Z_ROW(40));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 6)]) | STZ_Z_ROW(48));
        AMX_STZ(AMX_PTR(&t2[32 * (i + 7)]) | STZ_Z_ROW(56));
    }

    for (; i < 2 * nh; i++) {
        AMX_LDZ(AMX_PTR(&t2[32 * i]) | LDZ_Z_ROW(0));
        AMX_LDX(AMX_PTR(&t1[32 * i]) | LDX_REG(0));
        AMX_LDY(AMX_PTR(&t1[32 * (i + 2 * nl)]) | LDY_REG(0));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);

        AMX_STZ(AMX_PTR(&t2[32 * i]) | STZ_Z_ROW(0));
    }

    if (nl > nh) {
        AMX_LDZ(AMX_PTR(&t2[32 * i]) | LDZ_Z_ROW(0));
        AMX_LDX(AMX_PTR(&t1[32 * i]) | LDX_REG(0));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ENABLE_MODE(0) |
                   VECINT_ENABLE_VALUE(5) | VECINT_ALU_MODE_Z_SUB_X_SUB_Y);

        AMX_STZ(AMX_PTR(&t2[32 * i]) | STZ_Z_ROW(0));
    }

    // Add t2*x^(32 * nl) to t1 while reducing modulo x^d - 1 and saving to xy

    for (i = 0; i < 8 * (nl / 8); i += 8) {
        AMX_LDX_QUAD(&t1[32 * i], 0);
        AMX_LDX_QUAD(&t1[32 * (i + 4)], 4);

        AMX_LDY_QUAD(&t1[32 * i + d], 0);
        AMX_LDY_QUAD(&t1[32 * (i + 4) + d], 4);

        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl) + d]) | LDZ_Z_ROW(0));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 1) + d]) | LDZ_Z_ROW(8));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 2) + d]) | LDZ_Z_ROW(16));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 3) + d]) | LDZ_Z_ROW(24));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 4) + d]) | LDZ_Z_ROW(32));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 5) + d]) | LDZ_Z_ROW(40));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 6) + d]) | LDZ_Z_ROW(48));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl + 7) + d]) | LDZ_Z_ROW(56));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(1) | VECINT_Y_REG(1) | VECINT_Z_ROW(8) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(2) | VECINT_Y_REG(2) | VECINT_Z_ROW(16) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(3) | VECINT_Y_REG(3) | VECINT_Z_ROW(24) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(4) | VECINT_Y_REG(4) | VECINT_Z_ROW(32) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(5) | VECINT_Y_REG(5) | VECINT_Z_ROW(40) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(6) | VECINT_Y_REG(6) | VECINT_Z_ROW(48) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(7) | VECINT_Y_REG(7) | VECINT_Z_ROW(56) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_STZ(AMX_PTR(&xy[32 * i]) | STZ_Z_ROW(0));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 1)]) | STZ_Z_ROW(8));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 2)]) | STZ_Z_ROW(16));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 3)]) | STZ_Z_ROW(24));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 4)]) | STZ_Z_ROW(32));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 5)]) | STZ_Z_ROW(40));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 6)]) | STZ_Z_ROW(48));
        AMX_STZ(AMX_PTR(&xy[32 * (i + 7)]) | STZ_Z_ROW(56));
    }

    for (; i < nl; i++) {
        AMX_LDX(AMX_PTR(&t1[32 * i]) | LDX_REG(0));
        AMX_LDY(AMX_PTR(&t1[32 * i + d]) | LDY_REG(0));
        AMX_LDZ(AMX_PTR(&t2[32 * (i - nl) + d]) | LDZ_Z_ROW(0));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_STZ(AMX_PTR(&xy[32 * i]) | STZ_Z_ROW(0));
    }

    for (i = 0; i < 8 * (nh / 8); i += 8) {
        AMX_LDX_QUAD(&t1[32 * (i + nl)], 0);
        AMX_LDX_QUAD(&t1[32 * (i + nl + 4)], 4);

        AMX_LDY_QUAD(&t1[32 * (i + nl) + d], 0);
        AMX_LDY_QUAD(&t1[32 * (i + nl + 4) + d], 4);

        AMX_LDZ(AMX_PTR(&t2[32 * i]) | LDZ_Z_ROW(0));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 1)]) | LDZ_Z_ROW(8));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 2)]) | LDZ_Z_ROW(16));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 3)]) | LDZ_Z_ROW(24));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 4)]) | LDZ_Z_ROW(32));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 5)]) | LDZ_Z_ROW(40));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 6)]) | LDZ_Z_ROW(48));
        AMX_LDZ(AMX_PTR(&t2[32 * (i + 7)]) | LDZ_Z_ROW(56));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(1) | VECINT_Y_REG(1) | VECINT_Z_ROW(8) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(2) | VECINT_Y_REG(2) | VECINT_Z_ROW(16) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(3) | VECINT_Y_REG(3) | VECINT_Z_ROW(24) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(4) | VECINT_Y_REG(4) | VECINT_Z_ROW(32) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(5) | VECINT_Y_REG(5) | VECINT_Z_ROW(40) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(6) | VECINT_Y_REG(6) | VECINT_Z_ROW(48) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_REG(7) | VECINT_Y_REG(7) | VECINT_Z_ROW(56) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_STZ(AMX_PTR(&xy[32 * (i + nl)]) | STZ_Z_ROW(0));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 1)]) | STZ_Z_ROW(8));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 2)]) | STZ_Z_ROW(16));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 3)]) | STZ_Z_ROW(24));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 4)]) | STZ_Z_ROW(32));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 5)]) | STZ_Z_ROW(40));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 6)]) | STZ_Z_ROW(48));
        AMX_STZ(AMX_PTR(&xy[32 * (i + nl + 7)]) | STZ_Z_ROW(56));
    }

    for (; i < nh; i++) {
        AMX_LDX(AMX_PTR(&t1[32 * (i + nl)]) | LDX_REG(0));
        AMX_LDY(AMX_PTR(&t1[32 * (i + nl) + d]) | LDY_REG(0));
        AMX_LDZ(AMX_PTR(&t2[32 * i]) | LDZ_Z_ROW(0));

        AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_STZ(AMX_PTR(&xy[32 * (i + nl)]) | STZ_Z_ROW(0));
    }
}
