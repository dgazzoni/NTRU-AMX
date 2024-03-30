#ifndef AMX_H
#define AMX_H

#include "aarch64.h"

#define AMX_PTR(ptr) ((uint64_t)(ptr))

// Loads
// Default arguments for load instructions: LDX_LOAD_SINGLE, LDY_LOAD_SINGLE, LDZ_LOAD_SINGLE

// ldx

#define LDX_REG(reg) ((uint64_t)(reg) << 56)

#define LDX_LOAD_SINGLE (0ULL << 62)
#define LDX_LOAD_PAIR (1ULL << 62)

// ldy

#define LDY_REG(reg) ((uint64_t)(reg) << 56)

#define LDY_LOAD_SINGLE (0ULL << 62)
#define LDY_LOAD_PAIR (1ULL << 62)

// ldz

#define LDZ_Z_ROW(row) ((uint64_t)(row) << 56)

#define LDZ_LOAD_SINGLE (0ULL << 62)
#define LDZ_LOAD_PAIR (1ULL << 62)

// Stores
// Defalt arguments: STX_STORE_SINGLE, STY_STORE_SINGLE, STZ_STORE_SINGLE

// stx

#define STX_REG(reg) ((uint64_t)(reg) << 56)

#define STX_STORE_SINGLE (0ULL << 62)
#define STX_STORE_PAIR (1ULL << 62)

// sty

#define STY_REG(reg) ((uint64_t)(reg) << 56)

#define STY_STORE_SINGLE (0ULL << 62)
#define STY_STORE_PAIR (1ULL << 62)

// stz

#define STZ_Z_ROW(row) ((uint64_t)(row) << 56)

#define STZ_STORE_SINGLE (0ULL << 62)
#define STZ_STORE_PAIR (1ULL << 62)

// mac16
// Default arguments: MAC16_MATRIX, MAC16_X_NO_SKIP, MAC16_Y_NO_SKIP, MAC16_Z_NO_SKIP, MAC16_X_ENABLE_MODE(0),
//                    MAC16_X_ENABLE_VALUE(0), MAC16_MATRIX_Y_ENABLE_MODE(0), MAC16_MATRIX_Y_ENABLE_VALUE(0)

#define MAC16_MATRIX (0ULL << 63)
#define MAC16_VECTOR (1ULL << 63)

// mac16 common

#define MAC16_X_OFFSET(bytes) ((uint64_t)(bytes) << 10)
#define MAC16_X_REG(reg) (MAC16_X_OFFSET((reg)*64))

#define MAC16_Y_OFFSET(bytes) ((uint64_t)(bytes) << 0)
#define MAC16_Y_REG(reg) (MAC16_Y_OFFSET((reg)*64))

#define MAC16_Z_ROW(row) ((uint64_t)(row) << 20)

#define MAC16_Z_NO_SKIP (0ULL << 27)
#define MAC16_Z_SKIP (1ULL << 27)

#define MAC16_Y_NO_SKIP (0ULL << 28)
#define MAC16_Y_SKIP (1ULL << 28)

#define MAC16_X_NO_SKIP (0ULL << 29)
#define MAC16_X_SKIP (1ULL << 29)

#define MAC16_X_ENABLE_VALUE(value) ((uint64_t)(value) << 41)

#define MAC16_X_ENABLE_MODE(mode) ((uint64_t)(mode) << 46)

// mac16 matrix mode (bit 63 = 0)

#define MAC16_MATRIX_Y_ENABLE_VALUE(value) ((uint64_t)(value) << 32)

#define MAC16_MATRIX_Y_ENABLE_MODE(mode) ((uint64_t)(mode) << 37)

// vecint
// Default arguments: VECINT_ALU_MODE_Z_ADD_X_MUL_Y

#define VECINT_X_OFFSET(bytes) ((uint64_t)(bytes) << 10)
#define VECINT_X_REG(reg) (VECINT_X_OFFSET((reg)*64))

#define VECINT_Y_OFFSET(bytes) ((uint64_t)(bytes) << 0)
#define VECINT_Y_REG(reg) (VECINT_Y_OFFSET((reg)*64))

#define VECINT_Z_ROW(row) ((uint64_t)(row) << 20)

#define VECINT_ALU_MODE_Z_ADD_X_MUL_Y (0ULL << 47)
#define VECINT_ALU_MODE_Z_ADD_X_ADD_Y (2ULL << 47)
#define VECINT_ALU_MODE_Z_SUB_X_SUB_Y (3ULL << 47)

#define VECINT_ENABLE_VALUE(value) ((uint64_t)(value) << 32)

#define VECINT_ENABLE_MODE(mode) ((uint64_t)(mode) << 38)

// extrh

// extrh not defined in aarch64.h
#define AMX_EXTRH(gpr) AMX_OP_GPR(8, gpr)

#define EXTRH_COPY_ONLY (0ULL << 26)
#define EXTRH_GENERIC (1ULL << 26)

// extrh copy-only mode (bit 26 = 0)

#define EXTRH_COPY_ONLY_OFFSET(offset) ((uint64_t)(offset) << 10)
#define EXTRH_COPY_ONLY_REG(reg) EXTRH_COPY_ONLY_OFFSET((reg)*64)

#define EXTRH_COPY_ONLY_Z_ROW(row) ((uint64_t)(row) << 20)

#define EXTRH_COPY_ONLY_LANE_WIDTH_16_BIT (2ULL << 28)

// extrh generic mode (bit 26 = 1)
// Default arguments for extrh generic: EXTRH_DESTINATION_X

#define EXTRH_OFFSET(offset) ((uint64_t)(offset) << 0)
#define EXTRH_REG(reg) EXTRH_OFFSET((reg)*64)

#define EXTRH_DESTINATION_X (0ULL << 10)
#define EXTRH_DESTINATION_Y (1ULL << 10)

#define EXTRH_Z_ROW(row) ((uint64_t)(row) << 20)

#define EXTRH_LANE_WIDTH_16_BIT (1ULL << 11)

#endif  // AMX_H
