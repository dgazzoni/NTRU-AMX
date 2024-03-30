#ifndef POLYMODMUL_KARATSUBA_H
#define POLYMODMUL_KARATSUBA_H

#include <stdint.h>

void amx_karatsuba_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[],
                                                                         int d, int n);

#endif
