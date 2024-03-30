#include "aarch64.h"
#include "amx.h"

// extrv

// extrv not defined in aarch64.h
#define AMX_EXTRV(gpr) AMX_OP_GPR(9, gpr)

#define EXTRV_COPY_ONLY (0ULL << 26)

#define EXTRV_COPY_ONLY_OFFSET(offset) ((uint64_t)(offset))
#define EXTRV_COPY_ONLY_REG(reg) EXTRV_COPY_ONLY_OFFSET((reg)*64)

#define EXTRV_COPY_ONLY_Z_COLUMN(row) ((uint64_t)(row) << 20)

#define EXTRV_COPY_ONLY_LANE_WIDTH_16_BIT (2ULL << 28)

// We use this function to initialize AMX. The use of __attribute__((constructor)) ensures this function is called
// before main().
__attribute__((constructor)) void init_AMX(void) {
    AMX_SET();
}

void amx_latency_experiment(uint16_t *z, const uint16_t *x, const uint16_t *y, int iters) {
    AMX_LDX(AMX_PTR(x) | LDX_REG(0));
    AMX_LDY(AMX_PTR(y) | LDX_REG(0));

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    for (int i = 0; i < iters; i++) {
        AMX_EXTRH(EXTRH_COPY_ONLY | EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT | EXTRH_COPY_ONLY_REG(0) |
                  EXTRH_COPY_ONLY_Z_ROW(0));
        AMX_EXTRV(EXTRV_COPY_ONLY | EXTRV_COPY_ONLY_LANE_WIDTH_16_BIT | EXTRV_COPY_ONLY_REG(0) |
                  EXTRV_COPY_ONLY_Z_COLUMN(0));
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_NO_SKIP);
    }

    AMX_STZ(AMX_PTR(z) | STZ_Z_ROW(0));
}
