/* Based on supercop-20200702/crypto_core/invhrss701/simpler/core.c */

#include "poly.h"

#include <stdio.h>
#include <arm_neon.h>

#define BITARRAY_SIZE ((NTRU_N - 1 + 63) / 64)

/* return -1 if x<0 and y<0; otherwise return 0 */
static inline int16_t both_negative_mask(int16_t x, int16_t y) {
    return (x & y) >> 15;
}

void poly_R2_inv(poly *r, const poly *a) {

    uint64_t f[BITARRAY_SIZE];
    uint64_t g[BITARRAY_SIZE];
    uint64_t v[BITARRAY_SIZE];
    uint64_t w[BITARRAY_SIZE];
    uint64_t signx64, swapx64;
    uint64_t tx64;
    size_t i, loop;
    int16_t delta, sign, swap;

    for(i = 0; i < BITARRAY_SIZE; i++){
        v[i] = 0;
    }
    for(i = 1; i < BITARRAY_SIZE; i++){
        w[i] = 0;
    }
    w[0] = 1;
    for(i = 0; i < BITARRAY_SIZE - 1; i++){
        f[i] = 0xffffffffffffffff;
    }
    f[BITARRAY_SIZE - 1] = (1UL << (NTRU_N % 64)) - 1;
    for(i = 0; i < BITARRAY_SIZE; i++){
        g[i] = 0;
    }
    for(i = 0; i < NTRU_N - 1; i++){
        g[(NTRU_N - 2 - i) / 64] |= ((uint64_t)( (a->coeffs[i] ^ a->coeffs[NTRU_N - 1]) & 1)) << ((NTRU_N - 2 - i) % 64);
    }

    delta = 1;

    for (loop = 0; loop < 2 * (NTRU_N - 1) - 1; ++loop) {

        for(i = BITARRAY_SIZE - 1; i > 0; i--){
            v[i] = (v[i] << 1) | (v[i - 1] >> 63 );
        }
        v[0] <<= 1;

        sign = g[0] & f[0] & 1;
        swap = both_negative_mask(-delta, -(int16_t) (g[0] & 1));
        delta ^= swap & (delta ^ -delta);
        delta += 1;

        signx64 = (uint64_t) ((int64_t)(-sign));
        swapx64 = (uint64_t)((int64_t)swap);

        for(i = 0; i < BITARRAY_SIZE; i++){
            tx64 = swapx64 & (f[i] ^ g[i]);
            f[i] ^= tx64;
            g[i] ^= tx64;
            g[i] ^= signx64 & f[i];
        }
        for(i = 0; i < BITARRAY_SIZE; i++){
            tx64 = swapx64 & (v[i] ^ w[i]);
            v[i] ^= tx64;
            w[i] ^= tx64;
            w[i] ^= signx64 & v[i];
        }

        for(i = 0; i < BITARRAY_SIZE - 1; i++){
            g[i] = (g[i] >> 1) | (g[i + 1] << 63);
        }
        g[BITARRAY_SIZE - 1] >>= 1;

    }

    for (i = 0; i < NTRU_N - 1; ++i) {
        r->coeffs[i] = (v[(NTRU_N - 2 - i) / 64] >> ((NTRU_N - 2 - i) % 64) ) & 1;
    }
    r->coeffs[NTRU_N - 1] = 0;
}
