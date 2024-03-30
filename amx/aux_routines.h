#ifndef AUX_ROUTINES_H
#define AUX_ROUTINES_H

#include <stdint.h>

extern uint16_t *zeros;

void amx_poly_mul_mod_65536_u16_flatten_first_two_blocks(uint16_t xy[64], uint64_t xy_reg);

void amx_poly_mul_mod_65536_u16_flatten_middle_block(uint16_t xy[32], uint64_t xy_reg, uint64_t zi);

void amx_poly_mul_mod_65536_u16_flatten_last_two_blocks(uint16_t xy[64], uint64_t xy_reg);

void amx_poly_mul_mod_65536_u16_merge_first_and_last_blocks(uint16_t xy[32], uint64_t xy_reg, uint64_t zi,
                                                            uint64_t d_mod_32);

#endif  // AUX_ROUTINES_H
