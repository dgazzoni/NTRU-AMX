#ifndef TEST_AMX_H
#define TEST_AMX_H

extern "C" {
#include "aarch64.h"
#include "amx.h"
}
#include "test.h"

static inline void clobber_all_AMX_regs() {
    uint16_t x[256], y[256], z[2048];

    gen_random(x, 256);
    gen_random(y, 256);
    gen_random(z, 2048);

    for (uint64_t i = 0; i < 8; i++) {
        AMX_LDX((uint64_t)&x[32 * i] | LDX_REG(i));
        AMX_LDY((uint64_t)&y[32 * i] | LDY_REG(i));
    }

    for (uint64_t i = 0; i < 64; i++) {
        AMX_LDZ((uint64_t)&z[32 * i] | LDZ_Z_ROW(i));
    }
}

#endif  // TEST_AMX_H
