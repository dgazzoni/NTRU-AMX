#include <arm_neon.h>
#include "poly.h"

#include <stddef.h>

void poly_lift(poly *r, const poly *a) {

    for(size_t i = 0; i < NTRU_N; i++){
        r->coeffs[i] = a->coeffs[i];
    }
    poly_Z3_to_Zq(r);

}


