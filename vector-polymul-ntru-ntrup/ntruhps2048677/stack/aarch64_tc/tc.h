#ifndef TC_H
#define TC_H

#include <stdint.h>

#include "params.h"

// ensure TC_POLY_N <= POLY_N
#define TC_POLY_N 720

#define SB0 (TC_POLY_N / 5) // 144
#define SB1 (SB0 / 3)        // 48
#define SB2 (SB1 / 3)        // 16

#define SB2_RES (2 * SB2) // 32  = 16*2, 32/16 = 2
#define SB1_RES (2 * SB1) // 96  = 48*2, 96/16 = 6
#define SB0_RES (2 * SB0) // 288 = 144*2, 288/16 = 18

#define MASK (NTRU_Q - 1)

#define inv3 43691
#define inv5 52429
#define inv7 28087
#define inv9 36409
#define inv15 61167

#define inv49 22737
#define inv35 44939
#define inv45 20389
#define inv75 51555
#define inv525 7365
#define inv105 36825
#define inv225 17185
#define inv315 12275
#define inv25 23593

void schoolbook_16x16(uint16_t r[2 * 16], const uint16_t a[16], const uint16_t b[16]);

void tc5(uint16_t *restrict w[9], uint16_t *restrict polynomial);
void tc33_mul(uint16_t *restrict polyC[9], uint16_t *restrict polyA[9], uint16_t *restrict polyB[9]);
void tc33(uint16_t *restrict w, uint16_t *restrict src);
void k2(uint16_t *restrict w, uint16_t *restrict src);
void ik2(uint16_t *restrict w, uint16_t *restrict src);
void itc33(uint16_t *restrict dst, uint16_t *restrict w);
void itc5(uint16_t *restrict polynomial, uint16_t *w[9]);


void poly_mul_neon(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB);

#endif
