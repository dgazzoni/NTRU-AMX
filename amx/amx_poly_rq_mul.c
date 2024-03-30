#include <string.h>

#include "params.h"
#include "poly.h"
#include "polymodmul.h"

void poly_Rq_mul(poly *r, poly *a, poly *b) {
    amx_poly_mul_mod_65536_mod_x_d_minus_1_u16_32nx32n_coeffs(r->coeffs, a->coeffs, b->coeffs, NTRU_N,
                                                              (NTRU_N + 31) / 32);
}
