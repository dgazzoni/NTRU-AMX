#include "aux_routines.h"

#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include "aarch64.h"
#include "amx.h"

void init_AMX(void);

// We use this function to initialize AMX. The use of __attribute__((constructor)) ensures this function is called
// before main().
__attribute__((constructor)) void init_AMX(void) {
    AMX_SET();
}

void alloc_zeros(void);

uint16_t *zeros;

// We allocate and initialize a 32-element array of zeros, again before main()
__attribute__((constructor)) void alloc_zeros(void) {
    zeros = (uint16_t *)mmap(NULL, 32 * sizeof(uint16_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(zeros, 0, 32 * sizeof(uint16_t));
}

// This file contains algorithms 4.3, 4.4, 4.5 and 4.8 of the paper. We note, however, that register indices differ
// from those used in the paper; in fact the functions are parameterized by register index. This is due to an
// optimization described in the other files, which involves preloading slices to unused registers to avoid replaying
// loads.

// This is Algorithm 4.3 in the paper.
void amx_poly_mul_mod_65536_u16_flatten_first_two_blocks(uint16_t xy[64], uint64_t xy_reg) {
    const uint64_t extrh_x_args = EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT;
    const uint64_t extrh_y_args = EXTRH_GENERIC | EXTRH_DESTINATION_Y | EXTRH_LANE_WIDTH_16_BIT;

    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(2));
    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 2) | EXTRH_COPY_ONLY_Z_ROW(3));

    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 1) - 2) | MAC16_Z_ROW(0) | MAC16_Y_SKIP);
    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 2) - 2) | MAC16_Z_ROW(1) | MAC16_Y_SKIP);

    for (uint64_t i = 4; i < 64; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(i));
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 2) | EXTRH_COPY_ONLY_Z_ROW(i + 1));

        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg + 1) | EXTRH_Z_ROW(i + 2));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg + 2) | EXTRH_Z_ROW(i + 3));

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 1) - i) | VECINT_Y_OFFSET(64 * (xy_reg + 1) - i - 2) |
                   VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 2) - i) | VECINT_Y_OFFSET(64 * (xy_reg + 2) - i - 2) |
                   VECINT_Z_ROW(1) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    AMX_STZ(AMX_PTR(xy) | STZ_Z_ROW(0));
    AMX_STZ(AMX_PTR(&xy[32]) | STZ_Z_ROW(1));
}

// This is Algorithm 4.4 in the paper.
void amx_poly_mul_mod_65536_u16_flatten_middle_block(uint16_t xy[32], uint64_t xy_reg, uint64_t r) {
    const uint64_t extrh_x_args = EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT;
    const uint64_t extrh_y_args = EXTRH_GENERIC | EXTRH_DESTINATION_Y | EXTRH_LANE_WIDTH_16_BIT;

    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(3 - r));
    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(2 + r));

    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 1) - 2) | MAC16_Z_ROW(r) | MAC16_Y_SKIP);

    for (uint64_t i = 4; i < 64; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(i + 1 - r));
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(i + r));

        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg) | EXTRH_Z_ROW(i + 3 - r));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg + 1) | EXTRH_Z_ROW(i + 2 + r));

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 1) - i) | VECINT_Y_OFFSET(64 * (xy_reg + 1) - i - 2) |
                   VECINT_Z_ROW(r) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    AMX_STZ(AMX_PTR(xy) | STZ_Z_ROW(r));
}

// This is Algorithm 4.5 in the paper.
void amx_poly_mul_mod_65536_u16_flatten_last_two_blocks(uint16_t xy[64], uint64_t xy_reg) {
    const uint64_t extrh_x_args = EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT;
    const uint64_t extrh_y_args = EXTRH_GENERIC | EXTRH_DESTINATION_Y | EXTRH_LANE_WIDTH_16_BIT;

    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(3));
    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(2));

    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 1) - 2) | MAC16_Z_ROW(0) | MAC16_Y_SKIP);
    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 2) - 2) | MAC16_Z_ROW(1) | MAC16_Y_SKIP | MAC16_Z_SKIP);

    for (uint64_t i = 4; i < 64; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(i + 1));
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(i));

        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg) | EXTRH_Z_ROW(i + 3));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg + 1) | EXTRH_Z_ROW(i + 2));

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 1) - i) | VECINT_Y_OFFSET(64 * (xy_reg + 1) - i - 2) |
                   VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 2) - i) | VECINT_Y_OFFSET(64 * (xy_reg + 2) - i - 2) |
                   VECINT_Z_ROW(1) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    AMX_STZ(AMX_PTR(xy) | STZ_Z_ROW(0));
    AMX_STZ(AMX_PTR(&xy[32]) | STZ_Z_ROW(1));
}

// This is Algorithm 4.8 in the paper.
void amx_poly_mul_mod_65536_u16_merge_first_and_last_blocks(uint16_t xy[32], uint64_t xy_reg, uint64_t r,
                                                            uint64_t d_mod_32) {
    const uint64_t extrh_x_args = EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT;
    const uint64_t extrh_y_args = EXTRH_GENERIC | EXTRH_DESTINATION_Y | EXTRH_LANE_WIDTH_16_BIT;

    uint64_t offset = 2 * (32 - d_mod_32), i;

    AMX_LDZ(AMX_PTR(xy) | LDZ_Z_ROW(r));

    AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(3 - r));

    AMX_MAC16(MAC16_VECTOR | MAC16_X_OFFSET(64 * (xy_reg + 1) - offset - 2) | MAC16_Z_ROW(r) | MAC16_Y_SKIP);

    for (i = 2; i < 62 - offset; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(i + 3 - r));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg) | EXTRH_Z_ROW(i + 5 - r));

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 1) - offset - i - 2) |
                   VECINT_Y_OFFSET(64 * (xy_reg + 1) - offset - i - 4) | VECINT_Z_ROW(r) |
                   VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    for (; i < 62; i += 4) {
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg) | EXTRH_COPY_ONLY_Z_ROW(i + 2 + r));
        AMX_EXTRH(extrh_x_args | EXTRH_COPY_ONLY_REG(xy_reg + 1) | EXTRH_COPY_ONLY_Z_ROW(i + 3 - r));

        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg) | EXTRH_Z_ROW(i + 4 + r));
        AMX_EXTRH(extrh_y_args | EXTRH_REG(xy_reg + 1) | EXTRH_Z_ROW(i + 5 - r));

        AMX_VECINT(VECINT_X_OFFSET(64 * (xy_reg + 2) - offset - i - 2) |
                   VECINT_Y_OFFSET(64 * (xy_reg + 2) - offset - i - 4) | VECINT_Z_ROW(r) |
                   VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
    }

    AMX_STZ(AMX_PTR(xy) | STZ_Z_ROW(r));
}
