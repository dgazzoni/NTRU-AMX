#include "polymodmul.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "aarch64.h"
#include "amx.h"
#include "aux_routines.h"

// This is Algorithm 4.9 in the paper; however, d here corresponds to n in the paper, while the optimal choice for n is
// the next multiple of 32, i.e. 32*ceil(d/32).
void amx_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[], int d,
                                                               int n) {
    uint64_t mac16_matrix_extra_args = MAC16_MATRIX_Y_ENABLE_MODE(2) | MAC16_MATRIX_Y_ENABLE_VALUE(d % 32);

    int i, j, parity, reg_X, reg_Y;
    int d32 = ((d % 32) == 0) ? 32 : (d % 32);

    // We perform an optimization that is not described in the paper: we load slices of the input polynomials to unused
    // X and Y registers at the start of the function, so as to avoid replaying these loads every time they're needed.
    // Specifically: registers X_i and Y_i, for 0 <= i <= 5, are loaded with x[32*i : 32*i + 31] and
    // y[32*i : 32*i + 31]. Due to this, register indices in general do not match those of the paper.

    for (i = 0; i < 6; i++) {
        AMX_LDX(AMX_PTR(&x[32 * i]) | LDX_REG(i));
        AMX_LDY(AMX_PTR(&y[32 * i]) | LDY_REG(i));
    }

    // First block

    // The code does not employ explicit functions for the paper's Algorithm 4.7; instead, they are inlined into this
    // function. This block of code in particular corresponds to line 3 of Algorithm 4.9, which calls
    // AccumulateOuterProductsReduction(a, b, 0, 0, n, 32).

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    // Recall that registers Y_1 to Y_5 have been preloaded with the expected polynomial slices. We use X_6 as a
    // temporary for loading non-preloaded slices of x[].
    for (j = 1; j < 6; j++) {
        AMX_LDX(AMX_PTR(&x[d - 32 * j]) | LDX_REG(6));

        // In the paper, we assume that input coefficients x[d + 1, ..., 32n - 1] and y[d + 1, ..., 32n - 1] are all
        // zero. We could enforce this restriction using e.g. memset() calls as required; however, to avoid this
        // overhead, we have opted to introduce another optimization, which is to mask out certain rows/columns of outer
        // product calculations. Note that mac16_matrix_extra_args, declared at the start of the function, masks out
        // some of the bottom rows of the outer product. This parameter is added for the case j == n - 1, thus masking
        // out the calculation precisely when using coefficients y[d + 1, ..., 32n - 1].
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(6) | MAC16_Y_REG(j) | MAC16_Z_ROW(0) |
                  (j == n - 1 ? mac16_matrix_extra_args : 0));
    }

    // As we exhausted the preloaded slices of y[], we now load them explicitly.
    for (; j < n; j++) {
        AMX_LDX(AMX_PTR(&x[d - 32 * j]) | LDX_REG(6));
        AMX_LDY(AMX_PTR(&y[32 * j]) | LDY_REG(6));

        // See comment regarding mac16_matrix_extra_args above.
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(6) | MAC16_Y_REG(6) | MAC16_Z_ROW(0) |
                  (j == n - 1 ? mac16_matrix_extra_args : 0));
    }

    // Second block

    // We again inline Algorithm 4.7. This block corresponds to line 4 of Algorithm 4.9, which calls
    // AccumulateOuterProductsReduction(a, b, 1, 1, n, 32).
    //
    // Other comments from above equally apply to this block.

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(1) | MAC16_Y_REG(0) | MAC16_Z_ROW(1) | MAC16_Z_SKIP);
    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(1) | MAC16_Z_ROW(1));

    for (j = 2; j < 6; j++) {
        AMX_LDX(AMX_PTR(&x[d - 32 * (j - 1)]) | LDX_REG(6));

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(6) | MAC16_Y_REG(j) | MAC16_Z_ROW(1) |
                  (j == n - 1 ? mac16_matrix_extra_args : 0));
    }

    for (; j < n; j++) {
        AMX_LDX(AMX_PTR(&x[d - 32 * (j - 1)]) | LDX_REG(6));
        AMX_LDY(AMX_PTR(&y[32 * j]) | LDY_REG(6));

        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(6) | MAC16_Y_REG(6) | MAC16_Z_ROW(1) |
                  (j == n - 1 ? mac16_matrix_extra_args : 0));
    }

    // We load zeros to X_5 and Y_5 as required by Algorithm 4.3 (noting that we actually use different register indices
    // as compared to the paper). This displaces the preloaded slices in these registers, which are reloaded afterwards.
    AMX_LDX(AMX_PTR(zeros) | LDX_REG(5));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(5));

    amx_poly_mul_mod_65536_u16_flatten_first_two_blocks(xy, 5);

    AMX_LDX(AMX_PTR(&x[160]) | LDX_REG(5));
    AMX_LDY(AMX_PTR(&y[160]) | LDY_REG(5));

    // Third to (n - 1)-th block

    for (i = 2; i < n - 1; i++) {
        // The body of the loop is again an inlined version of Algorithm 4.7, realizing line 7 of Algorithm 4.9, which
        // calls AccumulateOuterProductsReduction(a, b, i, i mod 2, n, 32).
        //
        // Again, many of the comments from above apply here.
        parity = i % 2;

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

        for (j = 1; j <= i; j++) {
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

        for (j = i + 1; j < n; j++) {
            AMX_LDX(AMX_PTR(&x[d - 32 * (j - i)]) | LDX_REG(6));

            // Use the preloaded register if available, otherwise load the desired slice to Y_6.
            if (j < 6) {
                reg_Y = j;
            }
            else {
                AMX_LDY(AMX_PTR(&y[32 * j]) | LDY_REG(6));
                reg_Y = 6;
            }

            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(6) | MAC16_Y_REG(reg_Y) | MAC16_Z_ROW(parity) |
                      (j == n - 1 ? mac16_matrix_extra_args : 0));
        }

        amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);
    }

    // Final block

    // We again inline Algorithm 4.7. This block corresponds to line 9 of Algorithm 4.9, which calls
    // AccumulateOuterProductsReduction(a, b, n' − 1, (n' − 1) mod 2, n, n mod 32).
    //
    // Other comments from above equally apply to this block.

    parity = (n - 1) % 2;

    // Note from the call above that the parameter m to Algorithm 4.7 is n mod 32. The effect of this is preventing the
    // calculation of certain columns of the outer product. However, as this is the very first outer product
    // calculation of the block, which overwrites Z rather than accumulating with it, we need to initialize the complete
    // matrix (i.e. all columns of either all odd or even rows of Z, depending on parity). So, we use a trick here: we
    // load zeros to X_7, and then use a slice of X_6 || X_7 so that the exact desired number of zeroed columns are read
    // from X_7.
    AMX_LDX(AMX_PTR(&x[32 * (n - 2) + d32]) | LDX_REG(6));
    AMX_LDX(AMX_PTR(zeros) | LDX_REG(7));

    AMX_MAC16(MAC16_MATRIX | MAC16_X_OFFSET(7 * 64 - 2 * d32) | MAC16_Y_REG(0) | MAC16_Z_ROW(parity) | MAC16_Z_SKIP);

    for (j = 1; j < n - 1; j++) {
        // Use the preloaded register if available, otherwise load the desired slice to X_6.
        if (n - 1 - j < 6) {
            reg_X = n - 1 - j;
        }
        else {
            AMX_LDX(AMX_PTR(&x[32 * (n - 1 - j)]) | LDX_REG(6));
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

        // Note that some of the columns in the outer product are masked out by the use of MAC16_X_ENABLE_MODE(2),
        // corresponding to the "columns" parameter of the mac16 calls of Algorithm 4.7.
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(reg_X) | MAC16_Y_REG(reg_Y) | MAC16_Z_ROW(parity) |
                  MAC16_X_ENABLE_MODE(2) | MAC16_X_ENABLE_VALUE(d % 32));
    }

    // We use a similar trick as in the first mac16 call of the final block, but this time to avoid calculating the
    // bottom rows of the outer product, as well as some of the columns. This corresponds to bottom-left outer product
    // in the example of Figure 5(a) of the paper.
    AMX_LDY(AMX_PTR(&y[32 * (n - 2) + d32]) | LDY_REG(6));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(7));

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(n - 1 - j) | MAC16_Y_OFFSET(7 * 64 - 2 * d32) | MAC16_Z_ROW(parity) |
              MAC16_X_ENABLE_MODE(2) | MAC16_X_ENABLE_VALUE(d % 32));

    amx_poly_mul_mod_65536_u16_flatten_middle_block(&xy[32 * i], 6, parity);

    AMX_LDX(AMX_PTR(zeros) | LDX_REG(7));
    AMX_LDY(AMX_PTR(zeros) | LDY_REG(7));

    amx_poly_mul_mod_65536_u16_merge_first_and_last_blocks(xy, 6, 1 - parity, d32);
}
