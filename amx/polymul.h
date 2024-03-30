#ifndef POLYMUL_H
#define POLYMUL_H

#include <stdint.h>

void amx_poly_mul_mod_65536_u16_32x32_coeffs(uint16_t xy[64], uint16_t x[32], uint16_t y[32]);
void amx_poly_mul_mod_65536_u16_32nx32n_coeffs(uint16_t xy[], uint16_t x[], uint16_t y[], int n);

#endif  // POLYMUL_H
