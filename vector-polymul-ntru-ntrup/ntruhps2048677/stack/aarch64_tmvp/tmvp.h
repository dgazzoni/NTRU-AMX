#ifndef TMVP_H
#define TMVP_H

#include <stdint.h>
#include <stddef.h>

#include "poly.h"

// ensure TMVP_POLY_N <= POLY_N
#define TMVP_POLY_N 720

#define NTRU_N 677
#define SB0 (TMVP_POLY_N / 5) // 144
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

void tmvp(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB);

void ittc5(uint16_t *restrict w, uint16_t *restrict polynomial);
void tc5(uint16_t *restrict w, uint16_t *restrict polynomial);
void ttc5(uint16_t *restrict polynomial, uint16_t *restrict w);


void ittc3(uint16_t *restrict w, uint16_t *restrict src);
void ittc32(uint16_t *restrict w, uint16_t *restrict src);

void tc33(uint16_t *restrict w, uint16_t *restrict src);
void ttc33(uint16_t *restrict src, uint16_t *restrict w);

void tmvp33(uint16_t *restrict polyC, uint16_t *restrict toepA, uint16_t *restrict polyB);
void tmvp33_last(uint16_t *restrict polyC, uint16_t *restrict toepA, uint16_t *restrict polyB);

#define SIZE_L (18 * SB0)
#define SIZE_R (9 * SB0)
#define SIZE_I (9 * SB0)

#define F_L(des, src) ittc5(des, src)
#define F_R(des, src) tc5(des, src)
#define F_I(des, src) ttc5(des, src)
#define F_MUL(des, srcL, srcR) { \
    for(size_t i = 0; i < 8; i++){ \
        tmvp33(des + i * SB0, srcL + 2 * i * SB0, srcR + i * SB0); \
    } \
    tmvp33_last(des + 8 * SB0, srcL + 2 * 8 * SB0, srcR + 8 * SB0); \
}

#endif


