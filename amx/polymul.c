#include <stdint.h>

#include "aarch64.h"
#include "amx.h"
#include "aux_routines.h"

extern uint16_t *zeros;

// This is Algorithm 4.1 in the paper.
void amx_poly_mul_mod_65536_u16_32x32_coeffs(uint16_t xy[64], uint16_t x[32], uint16_t y[32]) {
    uint64_t extrh_x_args = EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT;
    uint64_t extrh_y_args = EXTRH_GENERIC | EXTRH_DESTINATION_Y | EXTRH_LANE_WIDTH_16_BIT;

    AMX_LDX(AMX_PTR(x) | LDX_REG(1));
    AMX_LDY(AMX_PTR(y) | LDY_REG(1));

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(0));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(0));

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(2));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(2));

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(1) | MAC16_Y_REG(1) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(1) | EXTRH_COPY_ONLY_Z_ROW(2));
    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 - 2) | MAC16_Z_ROW(0) | MAC16_Y_SKIP);
    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(128 - 2) | MAC16_Z_ROW(1) | MAC16_Y_SKIP | MAC16_Z_SKIP);

    for (uint64_t i = 4; i < 64; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(1) | EXTRH_COPY_ONLY_Z_ROW(i));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(1) | EXTRH_Z_ROW(i + 2));

        AMX_VECINT(VECINT_X_OFFSET(64 - i) | VECINT_Y_OFFSET(62 - i) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        AMX_VECINT(VECINT_X_OFFSET(128 - i) | VECINT_Y_OFFSET(126 - i) | VECINT_Z_ROW(1) |
                   VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    AMX_STZ(AMX_PTR(xy) | STZ_Z_ROW(0));
    AMX_STZ(AMX_PTR(&xy[32]) | STZ_Z_ROW(1));
}

// This is Algorithm 4.6 in the paper, for polynomial degrees 64, 96, ..., 192.
static void amx_poly_mul_mod_65536_u16_32nx32n_coeffs_small(uint16_t xy[], uint16_t x[], uint16_t y[], int n) {
    // We perform an optimization that is not described in the paper: we load slices of the input polynomials to unused
    // X and Y registers at the start of the function, so as to avoid replaying these loads every time they're needed.
    // Specifically: registers X_i and Y_i, for 0 <= i <= 5, are loaded with x[32*i : 32*i + 31] and
    // y[32*i : 32*i + 31]. Due to this, register indices in general do not match those of the paper.

    for (int i = 0; i < 4; i++) {
        AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(i));
        AMX_LDY(AMX_PTR(&y[32 * i]) | LDY_REG(i));
    }

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(5));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(5));

    // First block

    // The code does not employ explicit functions for the paper's Algorithm 4.2; instead, they are inlined into this
    // function. This block of code in particular corresponds to line 2 of Algorithm 4.6, which calls
    // AccumulateOuterProducts(a, b, 0, 0).

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    // Second block

    // We again inline Algorithm 4.2. This block corresponds to line 3 of Algorithm 4.6, which calls
    // AccumulateOuterProducts(a, b, 1, 1).

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(1) | MAC16_Y_REG(0) | MAC16_Z_ROW(1) | MAC16_Z_SKIP);
    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(1) | MAC16_Z_ROW(1));

    amx_poly_mul_mod_65536_u16_flatten_first_two_blocks(xy, 5);

    AMX_LDX(AMX_PTR(&x[128]) | LDX_REG(4));
    AMX_LDY(AMX_PTR(&y[128]) | LDY_REG(4));

    AMX_LDX(AMX_PTR(&x[160]) | LDX_REG(5));
    AMX_LDY(AMX_PTR(&y[160]) | LDY_REG(5));

    // Third to n-th block

    for (int i = 2; i < n; i++) {
        uint64_t parity = i % 2;

        // We again inline Algorithm 4.2. This block corresponds to line 6 of Algorithm 4.6, which calls
        // AccumulateOuterProducts(a, b, i, i mod 2).

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(i) | MAC16_Y_REG(0) | MAC16_Z_ROW(parity) | MAC16_Z_SKIP);

        for (int j = 1; j <= i; j++) {
            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(i - j) | MAC16_Y_REG(j) | MAC16_Z_ROW(parity));
        }

        amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);
    }

    // (n + 1)-th to (2 * n - 2)-th block

    for (int i = n; i < 2 * n - 2; i++) {
        uint64_t parity = i % 2;

        // We again inline Algorithm 4.2. This block corresponds to line 6 of Algorithm 4.6, which calls
        // AccumulateOuterProducts(a, b, i, i mod 2).

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(n - 1) | MAC16_Y_REG(i - n + 1) | MAC16_Z_ROW(parity) | MAC16_Z_SKIP);

        for (int j = i - n + 2; j < n; j++) {
            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(i - j) | MAC16_Y_REG(j) | MAC16_Z_ROW(parity));
        }

        amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);
    }

    // Final block

    // This corresponds to the call AccumulateOuterProducts(a, b, 2(n/32) − 2, 0) in line 8 of Algorithm 4.6.
    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(n - 1) | MAC16_Y_REG(n - 1) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(7));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(7));

    amx_poly_mul_mod_65536_u16_flatten_last_two_blocks(&xy[64 * (n - 1)], 5);
}

// This is Algorithm 4.6 in the paper. Degrees 32 and 64, 96, ..., 192 are treated specially, calling other functions,
// and the remainder of the code handles larger degrees.
void amx_poly_mul_mod_65536_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[], int n) {
    if (n == 1) {
        amx_poly_mul_mod_65536_u16_32x32_coeffs(xy, x, y);
        return;
    }
    else if (n <= 6) {
        amx_poly_mul_mod_65536_u16_32nx32n_coeffs_small(xy, x, y, n);
        return;
    }

    uint64_t reg_X, reg_Y;

    // We perform an optimization that is not described in the paper: we load slices of the input polynomials to unused
    // X and Y registers at the start of the function, so as to avoid replaying these loads every time they're needed.
    // Specifically: registers X_i and Y_i, for 0 <= i <= 5, are loaded with x[32*i : 32*i + 31] and
    // y[32*i : 32*i + 31]. Due to this, register indices in general do not match those of the paper.

    for (int i = 0; i < 4; i++) {
        AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(i));
        AMX_LDY(AMX_PTR(&y[32 * i]) | LDY_REG(i));
    }

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(5));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(5));

    // First block

    // The code does not employ explicit functions for the paper's Algorithm 4.2; instead, they are inlined into this
    // function. This block of code in particular corresponds to line 2 of Algorithm 4.6, which calls
    // AccumulateOuterProducts(a, b, 0, 0).

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    // Second block

    // We again inline Algorithm 4.2. This block corresponds to line 3 of Algorithm 4.6, which calls
    // AccumulateOuterProducts(a, b, 1, 1).

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(1) | MAC16_Z_ROW(1) | MAC16_Z_SKIP);
    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(1) | MAC16_Y_REG(0) | MAC16_Z_ROW(1));

    amx_poly_mul_mod_65536_u16_flatten_first_two_blocks(xy, 5);

    AMX_LDX(AMX_PTR(&x[128]) | LDX_REG(4));
    AMX_LDY(AMX_PTR(&y[128]) | LDY_REG(4));

    AMX_LDX(AMX_PTR(&x[160]) | LDX_REG(5));
    AMX_LDY(AMX_PTR(&y[160]) | LDY_REG(5));

    // Third to n-th block

    for (int i = 2; i < n; i++) {
        // The body of the loop is again an inlined version of Algorithm 4.2, realizing line 6 of Algorithm 4.6, which
        // calls AccumulateOuterProducts(a, b, i, i mod 2).

        uint64_t parity = i % 2;

        // If the slice we need has already been preloaded, we set reg_X to the index of the register containing it.
        // Otherwise (then-clause) we load it to X_6 and point reg_X to that register.
        if (i < 6) {
            reg_X = i;
        }
        else {
            AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(6));
            reg_X = 6;
        }

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(reg_X) | MAC16_Y_REG(0) | MAC16_Z_ROW(parity) | MAC16_Z_SKIP);

        for (int j = 1; j <= i; j++) {
            // Use the preloaded register if available, otherwise load the desired slice to X_6.
            if (i - j < 6) {
                reg_X = i - j;
            }
            else {
                AMX_LDX(AMX_PTR(&x[32 * (i - j)]) | LDX_REG(6));
                reg_X = 6;
            }

            // Use the preloaded register if available, otherwise load the desired slice to Y_6.
            if (j < 6) {
                reg_Y = j;
            }
            else {
                AMX_LDY(AMX_PTR(&y[32 * j]) | LDY_REG(6));
                reg_Y = 6;
            }

            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(reg_X) | MAC16_Y_REG(reg_Y) | MAC16_Z_ROW(parity));
        }

        amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);
    }

    for (int i = 0; i < 6; i++) {
        // Load the last slices of x[] and y[] in X_5 and Y_5, the next to last slices in X_4 and Y_4, and so on, until
        // X_0 and Y_0.
        AMX_LDX(AMX_PTR(&x[32 * (n - 6 + i)]) | LDX_REG(i));
        AMX_LDY(AMX_PTR(&y[32 * (n - 6 + i)]) | LDY_REG(i));
    }

    // (n + 1)-th to (2 * n - 2)-th block

    for (int i = n; i < 2 * n - 2; i++) {
        uint64_t parity = i % 2;

        // We again inline Algorithm 4.2. This block corresponds to line 6 of Algorithm 4.6, which calls
        // AccumulateOuterProducts(a, b, i, i mod 2).

        // Use the preloaded register if available, otherwise load the desired slice to Y_6.
        if (i >= 2 * n - 7) {
            reg_Y = i - (2 * n - 7);
        }
        else {
            AMX_LDY(AMX_PTR(&y[32 * (i - n + 1)]) | LDY_REG(6));
            reg_Y = 6;
        }

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(5) | MAC16_Y_REG(reg_Y) | MAC16_Z_ROW(parity) | MAC16_Z_SKIP);

        for (int j = i - n + 2; j < n; j++) {
            // Use the preloaded register if available, otherwise load the desired slice to X_6.
            if (i - j >= n - 6) {
                reg_X = i - j - (n - 6);
            }
            else {
                AMX_LDX(AMX_PTR(&x[32 * (i - j)]) | LDX_REG(6));
                reg_X = 6;
            }

            // Use the preloaded register if available, otherwise load the desired slice to Y_6.
            if (j >= n - 6) {
                reg_Y = j - (n - 6);
            }
            else {
                AMX_LDY(AMX_PTR(&y[32 * j]) | LDY_REG(6));
                reg_Y = 6;
            }

            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(reg_X) | MAC16_Y_REG(reg_Y) | MAC16_Z_ROW(parity));
        }

        amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);
    }

    // Final block

    // This corresponds to the call AccumulateOuterProducts(a, b, 2(n/32) − 2, 0) in line 8 of Algorithm 4.6.

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(5) | MAC16_Y_REG(5) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(7));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(7));

    amx_poly_mul_mod_65536_u16_flatten_last_two_blocks(&xy[64 * (n - 1)], 5);
}
