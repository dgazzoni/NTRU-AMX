#ifndef POLYMODMUL_H
#define POLYMODMUL_H

#include <stdint.h>

void amx_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[], int d, int n);

#endif
