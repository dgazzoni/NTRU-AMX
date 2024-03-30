#include <stdio.h>
#include <arm_neon.h>
#include "params.h"
#include "poly.h"
#include "batch_multiplication.h"

#include "tc.h"

void schoolbook_16x16(uint16_t r[2 * 16], const uint16_t a[16], const uint16_t b[16]) {
    size_t i, j;
    for (j = 0; j < 16; j++) {
        r[j] = a[0] * (uint32_t)b[j];
    }
    for (i = 1; i < 16; i++) {
        for (j = 0; j < 16 - 1; j++) {
            r[i + j] += a[i] * (uint32_t)b[j];
        }
        r[i + 16 - 1] = a[i] * (uint32_t)b[16 - 1];
    }
    r[2 * 16 - 1] = 0;
}


void k2(uint16_t *restrict w, uint16_t *restrict src) {
    for(int i = 0; i < 224/4; i++) {
        uint16x8x4_t a0, a1, a2;
        a0.val[0] = vld1q_u16(&src[0]);     
        a0.val[1] = vld1q_u16(&src[0] + 8); 
        a0.val[2] = vld1q_u16(&src[0] + 16);
        a0.val[3] = vld1q_u16(&src[0] + 24);
        a2.val[0] = vld1q_u16(&src[32]);     
        a2.val[1] = vld1q_u16(&src[32] + 8); 
        a2.val[2] = vld1q_u16(&src[32] + 16);
        a2.val[3] = vld1q_u16(&src[32] + 24);

        a1.val[0] = vaddq_u16(a0.val[0], a0.val[1]);
        a1.val[1] = vaddq_u16(a0.val[2], a0.val[3]);
        a1.val[2] = vaddq_u16(a2.val[0], a2.val[1]);
        a1.val[3] = vaddq_u16(a2.val[2], a2.val[3]);

        vst1q_u16_x4(&w[0], a1);
        src += 16*4;
        w += 8*4;
    }
}


void ik2(uint16_t *restrict w, uint16_t *restrict src) {
    for(int i = 0; i < 224/2; i++) {
        uint16x8_t w0, w1, w2, w3;
        w0 = vld1q_u16(&w[0]);
        w1 = vld1q_u16(&w[8]);
        w2 = vld1q_u16(&w[16]);
        w3 = vld1q_u16(&w[24]);
        uint16x8_t p0, p1;
        p0 = vld1q_u16(&src[0]);
        p1 = vld1q_u16(&src[8]);

        p0 = vsubq_u16(p0, w0);
        p0 = vsubq_u16(p0, w2);
        p0 = vaddq_u16(p0, w1);

        p1 = vsubq_u16(p1, w1);
        p1 = vsubq_u16(p1, w3);
        p1 = vaddq_u16(p1, w2);

        vst1q_u16(&w[8], p0);
        vst1q_u16(&w[16], p1);

        uint16x8_t w01, w11, w21, w31;
        w01 = vld1q_u16(&w[0+32]);
        w11 = vld1q_u16(&w[8+32]);
        w21 = vld1q_u16(&w[16+32]);
        w31 = vld1q_u16(&w[24+32]);
        uint16x8_t p01, p11;
        p01 = vld1q_u16(&src[0+16]);
        p11 = vld1q_u16(&src[8+16]);

        p01 = vsubq_u16(p01, w01);
        p01 = vsubq_u16(p01, w21);
        p01 = vaddq_u16(p01, w11);

        p11 = vsubq_u16(p11, w11);
        p11 = vsubq_u16(p11, w31);
        p11 = vaddq_u16(p11, w21);

        vst1q_u16(&w[8+32], p01);
        vst1q_u16(&w[16+32], p11);

        src += 16*2;
        w += 32*2;
    }
}


void tc33(uint16_t *restrict w, uint16_t *restrict src) {
    uint16_t *c0 = &src[0*SB2],
             *c1 = &src[1*SB2],
             *c2 = &src[2*SB2],
             *c3 = &src[3*SB2],
             *c4 = &src[4*SB2],
             *c5 = &src[5*SB2],
             *c6 = &src[6*SB2],
             *c7 = &src[7*SB2],
             *c8 = &src[8*SB2],
             *w00 = &w[ 0*SB2],
             *w01 = &w[ 1*SB2],
             *w02 = &w[ 2*SB2],
             *w03 = &w[ 3*SB2],
             *w04 = &w[ 4*SB2],
             *w05 = &w[ 5*SB2],
             *w06 = &w[ 6*SB2],
             *w07 = &w[ 7*SB2],
             *w08 = &w[ 8*SB2],
             *w09 = &w[ 9*SB2],
             *w10 = &w[10*SB2],
             *w11 = &w[11*SB2],
             *w12 = &w[12*SB2],
             *w13 = &w[13*SB2],
             *w14 = &w[14*SB2],
             *w15 = &w[15*SB2],
             *w16 = &w[16*SB2],
             *w17 = &w[17*SB2],
             *w18 = &w[18*SB2],
             *w19 = &w[19*SB2],
             *w20 = &w[20*SB2],
             *w21 = &w[21*SB2],
             *w22 = &w[22*SB2],
             *w23 = &w[23*SB2],
             *w24 = &w[24*SB2];
    // Utilize 22 SIMD registers
    uint16x8_t a0, a1, a2, a3, a4, a5, a6, a7, a8, //9
               tmp0, tmp1, tmp2, tmp3, // 4
               s0, s1, s2, // 3
               e0, e1, e2, // 3
               t0, t1, t2; // 3
    for (uint16_t addr = 0; addr < SB2; addr+=8) {
        a0 = vld1q_u16(&c0[addr]);
        a1 = vld1q_u16(&c1[addr]);
        a2 = vld1q_u16(&c2[addr]);
        a3 = vld1q_u16(&c3[addr]);
        a4 = vld1q_u16(&c4[addr]);
        a5 = vld1q_u16(&c5[addr]);
        a6 = vld1q_u16(&c6[addr]);
        a7 = vld1q_u16(&c7[addr]);
        a8 = vld1q_u16(&c8[addr]);

        tmp0 = vaddq_u16(a2, a0);
        tmp1 = vaddq_u16(tmp0, a1);
        tmp2 = vsubq_u16(tmp0, a1);
        tmp3 = vaddq_u16(tmp2, a2);
        tmp3 = vshlq_n_u16(tmp3, 1);
        tmp3 = vsubq_u16(tmp3, a0);

        vst1q_u16(&w00[addr], a0);
        vst1q_u16(&w01[addr], tmp1);
        vst1q_u16(&w02[addr], tmp2);
        vst1q_u16(&w03[addr], tmp3);
        vst1q_u16(&w04[addr], a2);

        tmp0 = vaddq_u16(a8, a6);
        tmp1 = vaddq_u16(tmp0, a7);
        tmp2 = vsubq_u16(tmp0, a7);
        tmp3 = vaddq_u16(tmp2, a8);
        tmp3 = vshlq_n_u16(tmp3, 1);
        tmp3 = vsubq_u16(tmp3, a6);

        vst1q_u16(&w20[addr], a6);
        vst1q_u16(&w21[addr], tmp1);
        vst1q_u16(&w22[addr], tmp2);
        vst1q_u16(&w23[addr], tmp3);
        vst1q_u16(&w24[addr], a8);

        s0 = vaddq_u16(a0, a6);
        s1 = vaddq_u16(a1, a7);
        s2 = vaddq_u16(a2, a8);

        e0 = vaddq_u16(s0, a3);
        e1 = vaddq_u16(s1, a4);
        e2 = vaddq_u16(s2, a5);

        tmp0 = vaddq_u16(e2, e0);
        tmp1 = vaddq_u16(tmp0, e1);
        tmp2 = vsubq_u16(tmp0, e1);
        tmp3 = vaddq_u16(tmp2, e2);
        tmp3 = vshlq_n_u16(tmp3, 1);
        tmp3 = vsubq_u16(tmp3, e0);

        vst1q_u16(&w05[addr], e0);
        vst1q_u16(&w06[addr], tmp1);
        vst1q_u16(&w07[addr], tmp2);
        vst1q_u16(&w08[addr], tmp3);
        vst1q_u16(&w09[addr], e2);

        e0 = vsubq_u16(s0, a3);
        e1 = vsubq_u16(s1, a4);
        e2 = vsubq_u16(s2, a5);

        tmp0 = vaddq_u16(e2, e0);
        tmp1 = vaddq_u16(tmp0, e1);
        tmp2 = vsubq_u16(tmp0, e1);
        tmp3 = vaddq_u16(tmp2, e2);
        tmp3 = vshlq_n_u16(tmp3, 1);
        tmp3 = vsubq_u16(tmp3, e0);

        vst1q_u16(&w10[addr], e0);
        vst1q_u16(&w11[addr], tmp1);
        vst1q_u16(&w12[addr], tmp2);
        vst1q_u16(&w13[addr], tmp3);
        vst1q_u16(&w14[addr], e2);

        t0 = vshlq_n_u16(a6, 1);
        t1 = vshlq_n_u16(a7, 1);
        t2 = vshlq_n_u16(a8, 1);
        t0 = vsubq_u16(t0, a3);
        t1 = vsubq_u16(t1, a4);
        t2 = vsubq_u16(t2, a5);
        t0 = vshlq_n_u16(t0, 1);
        t1 = vshlq_n_u16(t1, 1);
        t2 = vshlq_n_u16(t2, 1);
        t0 = vaddq_u16(t0, a0);
        t1 = vaddq_u16(t1, a1);
        t2 = vaddq_u16(t2, a2);

        tmp0 = vaddq_u16(t2, t0);
        tmp1 = vaddq_u16(tmp0, t1);
        tmp2 = vsubq_u16(tmp0, t1);
        tmp3 = vaddq_u16(tmp2, t2);
        tmp3 = vshlq_n_u16(tmp3, 1);
        tmp3 = vsubq_u16(tmp3, t0);

        vst1q_u16(&w15[addr], t0);
        vst1q_u16(&w16[addr], tmp1);
        vst1q_u16(&w17[addr], tmp2);
        vst1q_u16(&w18[addr], tmp3);
        vst1q_u16(&w19[addr], t2);
    }
}


void itc33(uint16_t *restrict dst, uint16_t *restrict w) {
    uint16_t *w0_mem[5],
             *w1_mem[5],
             *w2_mem[5],
             *w3_mem[5],
             *w4_mem[5];

             w0_mem[0] = &w[0*SB2_RES+0*5*SB2_RES];
             w1_mem[0] = &w[1*SB2_RES+0*5*SB2_RES];
             w2_mem[0] = &w[2*SB2_RES+0*5*SB2_RES];
             w3_mem[0] = &w[3*SB2_RES+0*5*SB2_RES];
             w4_mem[0] = &w[4*SB2_RES+0*5*SB2_RES];

             w0_mem[1] = &w[0*SB2_RES+1*5*SB2_RES];
             w1_mem[1] = &w[1*SB2_RES+1*5*SB2_RES];
             w2_mem[1] = &w[2*SB2_RES+1*5*SB2_RES];
             w3_mem[1] = &w[3*SB2_RES+1*5*SB2_RES];
             w4_mem[1] = &w[4*SB2_RES+1*5*SB2_RES];

             w0_mem[2] = &w[0*SB2_RES+2*5*SB2_RES];
             w1_mem[2] = &w[1*SB2_RES+2*5*SB2_RES];
             w2_mem[2] = &w[2*SB2_RES+2*5*SB2_RES];
             w3_mem[2] = &w[3*SB2_RES+2*5*SB2_RES];
             w4_mem[2] = &w[4*SB2_RES+2*5*SB2_RES];

             w0_mem[3] = &w[0*SB2_RES+3*5*SB2_RES];
             w1_mem[3] = &w[1*SB2_RES+3*5*SB2_RES];
             w2_mem[3] = &w[2*SB2_RES+3*5*SB2_RES];
             w3_mem[3] = &w[3*SB2_RES+3*5*SB2_RES];
             w4_mem[3] = &w[4*SB2_RES+3*5*SB2_RES];

             w0_mem[4] = &w[0*SB2_RES+4*5*SB2_RES];
             w1_mem[4] = &w[1*SB2_RES+4*5*SB2_RES];
             w2_mem[4] = &w[2*SB2_RES+4*5*SB2_RES];
             w3_mem[4] = &w[3*SB2_RES+4*5*SB2_RES];
             w4_mem[4] = &w[4*SB2_RES+4*5*SB2_RES];

    // 33 SIMD registers
    uint16x8_t r0, r1, r2, r3, r4, // 4x1 = 4
               v1, v2, v3,  // 3x1 = 3
               c1, c2, c3, c11, c21, c31; // 6x1 = 6
    uint16x8_t A[6][5]; // 4x5 = 20

    for (uint16_t addr = 0; addr < SB2; addr+= 8) {
// k = 0
        A[0][0] = vld1q_u16(&w0_mem[0][addr]);
        A[0][1] = vld1q_u16(&w0_mem[1][addr]);
        A[0][2] = vld1q_u16(&w0_mem[2][addr]);
        A[0][3] = vld1q_u16(&w0_mem[3][addr]);
        A[0][4] = vld1q_u16(&w0_mem[4][addr]);

        // v3 = (A[0][3] - A[0][1])*inv3
        v3 = vsubq_u16(A[0][3], A[0][1]);
        v3 = vmulq_n_u16(v3, inv3);

        // v1 = (A[0][1] - A[0][2]) >> 1
        v1 = vsubq_u16(A[0][1], A[0][2]);
        v1 = vshrq_n_u16(v1, 1);

        // v2 = (A[0][2] - A[0][0])
        v2 = vsubq_u16(A[0][2], A[0][0]);

        // c2 = v2 + v1 - A[0][4]
        c2 = vaddq_u16(v2, v1);
        c2 = vsubq_u16(c2, A[0][4]);

        // c3 = (v2 - v3)>>1  + (A[0][4] << 1)
        v2 = vsubq_u16(v2, v3);
        v2 = vshrq_n_u16(v2, 1);
        v3 = vshlq_n_u16(A[0][4], 1);
        c3 = vaddq_u16(v2, v3);

        // c1 = v1 - c3
        c1 = vsubq_u16(v1, c3);

        vst1q_u16(&dst[addr+0*SB2 + 0*SB1], A[0][0]);
        vst1q_u16(&dst[addr+0*SB2 + 1*SB1], c1);
        vst1q_u16(&dst[addr+0*SB2 + 2*SB1], c2);
        vst1q_u16(&dst[addr+0*SB2 + 3*SB1], c3);
        vst1q_u16(&dst[addr+0*SB2 + 4*SB1], A[0][4]);

        for(int j = 0; j < 5; j++) {
        r1 = vld1q_u16(&w1_mem[j][addr]); // 1
        r2 = vld1q_u16(&w2_mem[j][addr]); // -1
        r3 = vld1q_u16(&w3_mem[j][addr]); // -2
        r4 = vld1q_u16(&w4_mem[j][addr]); // inf

        // v3 = (r3 - r1)*inv3
        v3 = vsubq_u16(r3, r1);
        v3 = vmulq_n_u16(v3, inv3);

        // v1 = (r1 - r2) >> 1
        v1 = vsubq_u16(r1, r2);
        v1 = vshrq_n_u16(v1, 1);

        // v2 = (r2 - A[0][j])
        v2 = vsubq_u16(r2, A[0][j]);

        // c2 = v2 + v1 - r4
        c2 = vaddq_u16(v2, v1);
        c2 = vsubq_u16(c2, r4);

        // c3 = (v2 - v3)>>1  + (r4 << 1)
        v2 = vsubq_u16(v2, v3);
        v2 = vshrq_n_u16(v2, 1);
        v3 = vshlq_n_u16(r4, 1);
        c3 = vaddq_u16(v2, v3);

        // c1 = v1 - c3
        c1 = vsubq_u16(v1, c3);

// 2nd part
        r0 = vld1q_u16(&w0_mem[j][addr+SB2]); // 0
        r1 = vld1q_u16(&w1_mem[j][addr+SB2]); // 1
        r2 = vld1q_u16(&w2_mem[j][addr+SB2]); // -1
        r3 = vld1q_u16(&w3_mem[j][addr+SB2]); // -2
        A[5][j] = vld1q_u16(&w4_mem[j][addr+SB2]); // inf

        // v3 = (r3 - r1)*inv3
        v3 = vsubq_u16(r3, r1);
        v3 = vmulq_n_u16(v3, inv3);

        // v1 = (r1 - r2) >> 1
        v1 = vsubq_u16(r1, r2);
        v1 = vshrq_n_u16(v1, 1);

        // v2 = (r2 - r0)
        v2 = vsubq_u16(r2, r0);

        // c21 = v2 + v1 - A[5][j]
        c21 = vaddq_u16(v2, v1);
        c21 = vsubq_u16(c21, A[5][j]);

        // c31 = (v2 - v3)>>1  + (A[5][j] << 1)
        v2 = vsubq_u16(v2, v3);
        v2 = vshrq_n_u16(v2, 1);
        v3 = vshlq_n_u16(A[5][j], 1);
        c31 = vaddq_u16(v2, v3);

        // c11 = v1 - c31
        c11 = vsubq_u16(v1, c31);

        r0 = vaddq_u16(c1, r0);
        A[1][j] = r0;

        c11 = vaddq_u16(c2, c11);
        A[2][j] = c11;

        c21 = vaddq_u16(c3, c21);
        A[3][j] = c21;

        c31 = vaddq_u16(r4, c31);
        A[4][j] = c31;
        }

        for(uint16_t k = 1; k < 3; k++) {
        r0 = A[k][0];
        r1 = A[k][1];
        r2 = A[k][2];
        r3 = A[k][3];
        r4 = A[k][4];

        // v3 = (r3 - r1)*inv3
        v3 = vsubq_u16(r3, r1);
        v3 = vmulq_n_u16(v3, inv3);

        // v1 = (r1 - r2) >> 1
        v1 = vsubq_u16(r1, r2);
        v1 = vshrq_n_u16(v1, 1);

        // v2 = (r2 - r0)
        v2 = vsubq_u16(r2, r0);

        // c2 = v2 + v1 - r4
        c2 = vaddq_u16(v2, v1);
        c2 = vsubq_u16(c2, r4);

        // c3 = (v2 - v3)>>1  + (r4 << 1)
        v2 = vsubq_u16(v2, v3);
        v2 = vshrq_n_u16(v2, 1);
        v3 = vshlq_n_u16(r4, 1);
        c3 = vaddq_u16(v2, v3);

        // c1 = v1 - c3
        c1 = vsubq_u16(v1, c3);
        
        vst1q_u16(&dst[addr+k*SB2 + 0*SB1], r0);
        vst1q_u16(&dst[addr+k*SB2 + 1*SB1], c1);
        vst1q_u16(&dst[addr+k*SB2 + 2*SB1], c2);
        vst1q_u16(&dst[addr+k*SB2 + 3*SB1], c3);
        vst1q_u16(&dst[addr+k*SB2 + 4*SB1], r4);
        }
        for(uint16_t k = 3; k < 6; k++) {
        r0 = A[k][0];
        r1 = A[k][1];
        r2 = A[k][2];
        r3 = A[k][3];
        r4 = A[k][4];

        // v3 = (r3 - r1)*inv3
        v3 = vsubq_u16(r3, r1);
        v3 = vmulq_n_u16(v3, inv3);

        // v1 = (r1 - r2) >> 1
        v1 = vsubq_u16(r1, r2);
        v1 = vshrq_n_u16(v1, 1);

        // v2 = (r2 - r0)
        v2 = vsubq_u16(r2, r0);

        // c2 = v2 + v1 - r4
        c2 = vaddq_u16(v2, v1);
        c2 = vsubq_u16(c2, r4);

        // c3 = (v2 - v3)>>1  + (r4 << 1)
        v2 = vsubq_u16(v2, v3);
        v2 = vshrq_n_u16(v2, 1);
        v3 = vshlq_n_u16(r4, 1);
        c3 = vaddq_u16(v2, v3);

        // c1 = v1 - c3
        c1 = vsubq_u16(v1, c3);

        v1 = vld1q_u16(&dst[addr+k*SB2 + 0*SB1]);
        r0 = vaddq_u16(v1, r0);
        vst1q_u16(&dst[addr+k*SB2 + 0*SB1], r0);

        v2 = vld1q_u16(&dst[addr+k*SB2 + 1*SB1]);
        c1 = vaddq_u16(v2, c1);
        vst1q_u16(&dst[addr+k*SB2 + 1*SB1], c1);

        v3 = vld1q_u16(&dst[addr+k*SB2 + 2*SB1]);
        c2 = vaddq_u16(v3, c2);
        vst1q_u16(&dst[addr+k*SB2 + 2*SB1], c2);

        v1 = vld1q_u16(&dst[addr+k*SB2 + 3*SB1]);
        c3 = vaddq_u16(v1, c3);
        vst1q_u16(&dst[addr+k*SB2 + 3*SB1], c3);

        vst1q_u16(&dst[addr+k*SB2 + 4*SB1], r4);
        }
    }
}


void tc5(uint16_t *restrict w[9], uint16_t *restrict polynomial) {
    uint16_t *w0_mem = w[0],
             *w1_mem = w[1],
             *w2_mem = w[2],
             *w3_mem = w[3],
             *w4_mem = w[4],
             *w5_mem = w[5],
             *w6_mem = w[6],
             *w7_mem = w[7],
             *w8_mem = w[8],
             *c0 = &polynomial[0*SB0],
             *c1 = &polynomial[1*SB0],
             *c2 = &polynomial[2*SB0],
             *c3 = &polynomial[3*SB0],
             *c4 = &polynomial[4*SB0];
    uint16x8_t r0, r1, r2, r3, r4, p0, p1, p_1, tp;
    uint16x8_t zero;
    zero = vmovq_n_u16(0);
    for (uint16_t addr = 0; addr < 8*13; addr+= 8) {
        r0 = vld1q_u16(&c0[addr]);
        r1 = vld1q_u16(&c1[addr]);
        r2 = vld1q_u16(&c2[addr]);
        r3 = vld1q_u16(&c3[addr]);
        r4 = vld1q_u16(&c4[addr]);

        p0 = vaddq_u16(r0, r2);  // p0  = r0 + r2
        p0 = vaddq_u16(p0, r4);  // p0  = p0 + r4 = r0 + r2 + r4
        tp = vaddq_u16(r1, r3);  // tp  = r1 + r3

        p1 = vaddq_u16(p0, tp);       // p1  = p0 + tp = r0 + r2 + r4 + r1 + r3
        p_1 = vsubq_u16(p0, tp);      // p_1 = p0 - tp = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w0_mem[addr], r0); // A(0)   = r0
        vst1q_u16(&w1_mem[addr], p1); // A(1)   = r0 + r2 + r4 + r1 + r3
        vst1q_u16(&w2_mem[addr], p_1);// A(-1)  = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w8_mem[addr], r4); // A(inf) = r4


        // deal w/ A(2), A(-2)
        p0 = vshlq_n_u16(r4, 2);  // p0 = (4) *(r4)
        p0 = vaddq_u16(p0, r2);   // p0 = (4) *(r4) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r4) + (4)*r2
        p0 = vaddq_u16(p0, r0);   // p0 = (16)*(r4) + (4)*r2 + r0

        tp = vshlq_n_u16(r3,  2); // tp = (4)*(r3)
        tp = vaddq_u16(tp, r1);   // tp = (4)*(r3) + r1
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r3) + (2)*r1

        p1 = vaddq_u16(p0, tp);       // p1  = p0 + tp = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        p_1 = vsubq_u16(p0, tp);      // p_1 = p0 - tp = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1
        vst1q_u16(&w3_mem[addr], p1); // A(2)    = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        vst1q_u16(&w4_mem[addr], p_1);// A(-2)   = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1


        // deal w/ A(3)
        p0 = vmulq_n_u16(r4, 9);  // p0 = (9) *(r4)
        p0 = vaddq_u16(p0, r2);   // p0 = (9) *(r4) + r2
        p0 = vmulq_n_u16(p0, 9);  // p0 = (81)*(r4) + (9)*r2
        p0 = vaddq_u16(p0, r0);   // p0 = (81)*(r4) + (9)*r2 + r0

        tp = vmulq_n_u16(r3, 9);   // tp = (9)*(r3)
        tp = vaddq_u16(tp, r1);    // tp = (9)*(r3) + r1
        tp = vmulq_n_u16(tp, 3);   // tp = (27)*(r3) + (3)*r1

        p1 = vaddq_u16(p0, tp);        // p1  = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1
        vst1q_u16(&w5_mem[addr], p1);  // A(3)    = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1


        // deal w/ A(1/2), A(-1/2)
        p0 = vshlq_n_u16(r0, 2);  // p0 = (4) *(r0)
        p0 = vaddq_u16(p0, r2);   // p0 = (4) *(r0) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r0) + (4)*r2
        p0 = vaddq_u16(p0, r4);   // p0 = (16)*(r0) + (4)*r2 + r4

        tp = vshlq_n_u16(r1,  2); // tp = (4)*(r1)
        tp = vaddq_u16(tp, r3);   // tp = (4)*(r1) + r3
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r1) + (2)*r3

        p1 = vaddq_u16(p0, tp);  // p1  = p0 + tp = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3

        vst1q_u16(&w6_mem[addr], p1);  // A(1/2)   = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        vst1q_u16(&w7_mem[addr], p_1); // A(-1/2)  = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3
    }
    for (uint16_t addr = 8*13; addr < SB0; addr+= 8) {
        r0 = vld1q_u16(&c0[addr]);
        r1 = vld1q_u16(&c1[addr]);
        r2 = vld1q_u16(&c2[addr]);
        r3 = vld1q_u16(&c3[addr]);
        // r4 = vld1q_u16(&c4[addr]);  //r4 = 0

        p0 = vaddq_u16(r0, r2);  // p0  = r0 + r2
        tp = vaddq_u16(r1, r3);  // tp  = r1 + r3

        p1 = vaddq_u16(p0, tp); // p1  = p0 + tp = r0 + r2 + r4 + r1 + r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w0_mem[addr], r0); // A(0)   = r0
        vst1q_u16(&w1_mem[addr], p1); // A(1)   = r0 + r2 + r4 + r1 + r3
        vst1q_u16(&w2_mem[addr], p_1);// A(-1)  = r0 + r2 + r4 - r1 - r3

        vst1q_u16(&w8_mem[addr], zero); // A(inf) = r4


        // deal w/ A(2), A(-2)
        p0 = vshlq_n_u16(r2, 2);  // p0 = (16)*(r4) + (4)*r2
        p0 = vaddq_u16(p0, r0);   // p0 = (16)*(r4) + (4)*r2 + r0

        tp = vshlq_n_u16(r3,  2); // tp = (4)*(r3)
        tp = vaddq_u16(tp, r1);   // tp = (4)*(r3) + r1
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r3) + (2)*r1

        p1 = vaddq_u16(p0, tp);       // p1  = p0 + tp = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        p_1 = vsubq_u16(p0, tp);      // p_1 = p0 - tp = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1
        vst1q_u16(&w3_mem[addr], p1); // A(2)    = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        vst1q_u16(&w4_mem[addr], p_1);// A(-2)   = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1


        // deal w/ A(3)
        p0 = vmulq_n_u16(r2, 9);   // p0 = (81)*(r4) + (9)*r2
        p0 = vaddq_u16(p0, r0);    // p0 = (81)*(r4) + (9)*r2 + r0

        tp = vmulq_n_u16(r3, 9);   // tp = (9)*(r3)
        tp = vaddq_u16(tp, r1);    // tp = (9)*(r3) + r1
        tp = vmulq_n_u16(tp, 3);   // tp = (27)*(r3) + (3)*r1

        p1 = vaddq_u16(p0, tp);        // p1  = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1
        vst1q_u16(&w5_mem[addr], p1);  // A(3)    = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1

        // deal w/ A(1/2), A(-1/2)
        p0 = vshlq_n_u16(r0, 2);  // p0 = (4) *(r0)
        p0 = vaddq_u16(p0, r2);   // p0 = (4) *(r0) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r0) + (4)*r2

        tp = vshlq_n_u16(r1,  2); // tp = (4)*(r1)
        tp = vaddq_u16(tp, r3);   // tp = (4)*(r1) + r3
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r1) + (2)*r3

        p1 = vaddq_u16(p0, tp);   // p1  = p0 + tp = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        p_1 = vsubq_u16(p0, tp);  // p_1 = p0 - tp = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3

        vst1q_u16(&w6_mem[addr], p1);  // A(1/2)   = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        vst1q_u16(&w7_mem[addr], p_1); // A(-1/2)  = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3
    }
}

void itc5(uint16_t *restrict polynomial, uint16_t *w[9]) {
    uint16_t *w0_mem = w[0],
             *w1_mem = w[1],
             *w2_mem = w[2],
             *w3_mem = w[3],
             *w4_mem = w[4],
             *w5_mem = w[5],
             *w6_mem = w[6],
             *w7_mem = w[7],
             *w8_mem = w[8];
    uint16x8_t r0, r1, r2, r3, r4, r5, r6, r7, r8,
                 t0, t1, t2, t3, t4, t5,
                 c1, c2, c3, c4, c5, c6, c7;

    for (uint16_t addr = 0; addr < SB0_RES/2; addr+= 8) { // 9 = 288/16, inst lines: almost 120
        r0 = vld1q_u16(&w0_mem[addr]); // C(0) = A(0)*B(0)
        r1 = vld1q_u16(&w1_mem[addr]);
        r2 = vld1q_u16(&w2_mem[addr]);
        r3 = vld1q_u16(&w3_mem[addr]);
        r4 = vld1q_u16(&w4_mem[addr]);
        r5 = vld1q_u16(&w5_mem[addr]);
        r6 = vld1q_u16(&w6_mem[addr]);
        r7 = vld1q_u16(&w7_mem[addr]);
        r8 = vld1q_u16(&w8_mem[addr]); // C(f) = A(f)*B(f)


        // deal w/ theta-6
        t0 = vaddq_u16(r1, r2);      // t0 = r1         + r2
        t0 = vmulq_n_u16(t0, inv9);  // t0 = (1/9)*r1   + (1/9)*r2 --->
        t0 = vshlq_n_u16(t0, 1);     // t0 = (2/9)*r1     + (2/9)*r2
        t1 = vaddq_u16(r3, r4);      // t1 = r3         + r4
        t1 = vmulq_n_u16(t1, inv45); // t1 = (1/45)*r3  + (1/45)*r4
        t2 = vaddq_u16(r6, r7);      // t2 = r6         + r7
        t2 = vmulq_n_u16(t2, inv45); // t2 = (1/45)*r6  + (1/45)*r7
        t3 = vmulq_n_u16(r8, 21);    // t3 = (21)*r8
        c6 = vshrq_n_u16(t2, 1);     // t4 = (1/90)*r6 + (1/90)*r7
        c6 = vsubq_u16(c6, t3);      // c6 = (1/90)*r6 + (1/90)*r7 - (21)*r8
        c6 = vshrq_n_u16(c6, 1);     // c6 = (1/180)*r6 + (1/180)*r7 - (21/2)*r8
        c6 = vaddq_u16(c6, t1);      // c6 = (1/45)*r3  + (1/45)*r4 + (1/180)*r6 + (1/180)*r7 - (21/2)*r8
        c6 = vshrq_n_u16(c6, 1);     // c6 = (1/90)*r3  + (1/90)*r4 + (1/360)*r6 + (1/360)*r7 - (21/4)*r8
        c6 = vsubq_u16(c6, t0);
        c6 = vsubq_u16(c6, r0);      // c6 = (-1)*r0    + (-2/9)*r1 + (-2/9)*r2  + (1/90)*r3  + (1/90)*r4 + (1/360)*r6 + (1/360)*r7 - (21/4)*r8 ---> c6 fin


        // deal w/ theta-2
        t4 = vmulq_n_u16(r0, 21);   // t4 = (21)*r0
        c2 = vshrq_n_u16(t1, 1);    // c2 = (1/90)*r3 + (1/90)*r4
        c2 = vsubq_u16(c2, t4);     // c2 = -(21)*r0 + (1/90)*r3 + (1/90)*r4
        c2 = vshrq_n_u16(c2, 1);    // c2 = -(21/2)*r0 + (1/180)*r3 + (1/180)*r4
        c2 = vaddq_u16(c2, t2);     // c2 = -(21/2)*r0 + (1/180)*r3 + (1/180)*r4 + (1/45)*r6  + (1/45)*r7
        c2 = vshrq_n_u16(c2, 1);    // c2 = -(21/4)*r0 + (1/360)*r3 + (1/360)*r4 + (1/90)*r6  + (1/90)*r7
        c2 = vsubq_u16(c2, t0);
        c2 = vsubq_u16(c2, r8);     // c2 = (-21/4)*r0 + (-2/9)*r1  + (-2/9)*r2  + (1/360)*r3  + (1/360)*r4 + (1/90)*r6 + (1/90)*r7 + (-1)*r8 ---> c2 fin


        // deal w/ theta-4
        t2 = vaddq_u16(r6, r7);     // t2 = r6 + r7, note: cannot inherit from (1/90), 3 bit loss already
        t2 = vaddq_u16(t2, r3);     // t2 = r3 + r6 + r7
        t2 = vaddq_u16(t2, r4);     // t2 = r3         + r4        + r6         + r7
        t2 = vmulq_n_u16(t2, inv9); // t2 = (1/9)*r3   + (1/9)*r4  + (1/9)*r6   + (1/9)*r7
        t0 = vmulq_n_u16(t0, 17);   // t0 = (34/9)*r1  + (34/9)*r2
        c4 = vshrq_n_u16(t2, 1);    // c4 = (1/18)*r3   + (1/18)*r4  + (1/18)*r6   + (1/18)*r7
        c4 = vsubq_u16(t0, c4);
        c4 = vaddq_u16(c4, t4);
        c4 = vaddq_u16(c4, t3);     // c4 = (21)*r0 + (34/9)*r1  + (34/9)*r2 - (1/18)*r3   - (1/18)*r4  - (1/18)*r6   - (1/18)*r7 + (21)*r8
        c4 = vshrq_n_u16(c4, 2);    // c4 = (21/4)*r0  + (17/18)*r1  + (17/18)*r2 - (1/72)*r3   - (1/72)*r4  - (1/72)*r6   - (1/72)*r7 + (21/2)*r8 ---> c4 fin


        // deal w/ theta-7 and theta-1
        t0 = vmulq_n_u16(r3, inv45);  // t0 = (1/15)*r3
        t1 = vmulq_n_u16(r4, inv225); // t1 = (1/75)*r4
        t4 = vaddq_u16(t0, t1);       // t4 = (1/15)*r3 + (1/75)*r4
        t0 = vsubq_u16(t0, t1);
        t0 = vmulq_n_u16(t0, 3);      // t0 = (1/15)*r3 + (-1/75)*r4
        c1 = vshrq_n_u16(t0, 1);      // c1 = (1/30)*r3 + (-1/150)*r4
        t0 = vmulq_n_u16(r6, inv225); // t0 = (1/225)*r6
        t1 = vmulq_n_u16(r7, inv315); // t1 = (1/315)*r7
        t2 = vsubq_u16(t0, t1);
        t2 = vmulq_n_u16(t2, 3);      // t2 = (1/75)*r6 - (1/105)*r7
        c7 = vaddq_u16(t0, t1);       // c7 = (1/225)*r6 + (1/315)*r7
        c7 = vshrq_n_u16(c7, 1);      // c7 = (1/450)*r6 + (1/630)*r7
        t0 = vmulq_n_u16(r2, inv3);   // t0 = (1/3)*r2
        t1 = vmulq_n_u16(r5, inv525); // t1 = (1/525)*r5
        c1 = vaddq_u16(c1, t0);
        c1 = vsubq_u16(c1, t1);
        c1 = vaddq_u16(c1, t2);       // c1 = (1/3)*r2 + (1/30)*r3 + (-1/150)*r4 - (1/525)*r5 + (1/75)*r6 - (1/105)*r7
        c1 = vshrq_n_u16(c1, 1);      // c1 = (1/6)*r2 + (1/60)*r3 + (-1/300)*r4 - (1/1050)*r5 + (1/150)*r6 - (1/210)*r7
        t0 = vmulq_n_u16(r2, inv9);   // t0 = (1/9)*r2
        c7 = vsubq_u16(t1, c7);       // c7 = (1/525)*r5 - (1/450)*r6 - (1/630)*r7
        c7 = vaddq_u16(c7, t0);
        c7 = vsubq_u16(c7, t4);       // c7 = (1/9)*r2 - (1/45)*r3 - (1/225)*r4 + (1/525)*r5 - (1/450)*r6 - (1/630)*r7
        c7 = vshrq_n_u16(c7, 1);      // c7 = (1/18)*r2 - (1/90)*r3 - (1/450)*r4 + (1/1050)*r5 - (1/900)*r6 - (1/1260)*r7
        t0 = vmulq_n_u16(r0, inv3);   // t0 = (1/3)*r0
        t1 = vmulq_n_u16(r8, 3);      // t1 = 3*r8
        t0 = vsubq_u16(t0, t1);       // t0 = (1/3)*r0 - 3*r8
        t1 = vmulq_n_u16(r1, inv3);   // t1 = (1/3)*r1
        t2 = vmulq_n_u16(r1, inv9);   // t2 = (1/9)*r1
        c1 = vsubq_u16(c1, t0);
        c1 = vsubq_u16(c1, t1);       // c1 = -(1/3)*r0 - (1/3)*r1 + (1/6)*r2 + (1/60)*r3 + (-1/300)*r4 - (1/1050)*r5 + (1/150)*r6 - (1/210)*r7 + 3*r8 ---> c1 fin
        c7 = vaddq_u16(c7, t0);
        c7 = vaddq_u16(c7, t2);       // c7 = (1/3)*r0 + (1/9)*r1 + (1/18)*r2 - (1/90)*r3 - (1/450)*r4 + (1/1050)*r5 - (1/900)*r6 - (1/1260)*r7 - 3*r8 ---> c7 fin


        // deal w/ theta-5 and theta-3
        t3 = vmulq_n_u16(r5, inv25);                 // t3 = (1/25)*r5
        t5 = vmulq_n_u16(r7, inv45);                 // t5 = (1/45)*r7
        t0 = vmulq_n_u16(r2, (uint16_t)(47*inv9));   // t0 = (47/9)*r2
        t1 = vmulq_n_u16(r3, (uint16_t)(31*inv45));  // t1 = (31/45)*r3
        t2 = vmulq_n_u16(r4, (uint16_t)(29*inv225)); // t2 = (29/225)*r4
        t4 = vmulq_n_u16(r6, (uint16_t)(23*inv225)); // t4 = (23/225)*r6
        c3 = vaddq_u16(t0, t1);                      // c3 = (47/9)*r2 + (31/45)*r3
        c3 = vsubq_u16(t2, c3);
        c3 = vaddq_u16(c3, t3);
        c3 = vsubq_u16(c3, t4);
        c3 = vaddq_u16(c3, t5);                      // c3 = -(47/9)*r2 - (31/45)*r3 + (29/225)*r4 + (1/25)*r5 - (23/225)*r6 + (1/45)*r7
        c3 = vshrq_n_u16(c3, 1);                     // c3 = -(47/18)*r2 - (31/90)*r3 + (29/450)*r4 + (1/50)*r5 - (23/450)*r6 + (1/90)*r7
        t0 = vmulq_n_u16(r2, (uint16_t)(5*inv9));    // t0 = (5/9)*r2
        t1 = vmulq_n_u16(r3, (uint16_t)(29*inv45));  // t1 = (29/45)*r3
        t2 = vmulq_n_u16(r4, (uint16_t)(19*inv225)); // t2 = (19/225)*r4
        t4 = vmulq_n_u16(r6, (uint16_t)(13*inv225)); // t4 = (13/225)*r6
        c5 = vsubq_u16(t1, t0);                      // c5 = -(5/9)*r2 + (29/45)*r3
        c5 = vsubq_u16(c5, t2);
        c5 = vsubq_u16(c5, t3);
        c5 = vaddq_u16(c5, t4);
        c5 = vaddq_u16(c5, t5);                      // c5 = -(5/9)*r2 + (29/45)*r3 - (19/225)*r4 - (1/25)*r5 + (13/225)*r6 + (1/45)*r7
        c5 = vshrq_n_u16(c5, 1);                     // c5 = -(5/18)*r2 + (29/90)*r3 - (19/450)*r4 - (1/50)*r5 + (13/450)*r6 + (1/90)*r7
        t0 = vmulq_n_u16(r0, 7);                     // t0 = 7*r0
        t3 = vmulq_n_u16(r1, (uint16_t)(29*inv9));   // t3 = (29/9)*r1
        t1 = vmulq_n_u16(r1, (uint16_t)(55*inv9));   // t1 = (55/9)*r1
        t2 = vmulq_n_u16(r8, 63);                    // t2 = 63*r8
        t0 = vsubq_u16(t0, t2);                      // t0 = 7*r0 - 63*r8
        c3 = vaddq_u16(c3, t0);
        c3 = vaddq_u16(c3, t1);                     // c3 = 7*r0 + (55/9)*r1 - (47/18)*r2 - (31/90)*r3 + (29/450)*r4 + (1/50)*r5 - (23/450)*r6 + (1/90)*r7 - 63*r8
        c3 = vshrq_n_u16(c3, 2);                    // c3 = (7/4)*r0 + (55/36)*r1 - (47/72)*r2 - (31/360)*r3 + (29/1800)*r4 + (1/200)*r5 - (23/1800)*r6 + (1/360)*r7 - (63/4)*r8 ---> c3 fin
        c5 = vsubq_u16(c5, t0);
        c5 = vsubq_u16(c5, t3);                     // c5 = -7*r0 - (29/9)*r1 - (5/18)*r2 + (29/90)*r3 - (19/450)*r4 - (1/50)*r5 + (13/450)*r6 + (1/90)*r7 + 63*r8
        c5 = vshrq_n_u16(c5, 2);                    // c5 = -(7/4)*r0 - (29/36)*r1 - (5/72)*r2 + (29/360)*r3 - (19/1800)*r4 - (1/200)*r5 + (13/1800)*r6 + (1/360)*r7 + (63/4)*r8 ---> c5 fin

        vst1q_u16(&polynomial[addr + 0*SB0], r0);
        vst1q_u16(&polynomial[addr + 1*SB0], c1);
        vst1q_u16(&polynomial[addr + 2*SB0], c2);
        vst1q_u16(&polynomial[addr + 3*SB0], c3);
        vst1q_u16(&polynomial[addr + 4*SB0], c4);
        vst1q_u16(&polynomial[addr + 5*SB0], c5);
        vst1q_u16(&polynomial[addr + 6*SB0], c6);
        vst1q_u16(&polynomial[addr + 7*SB0], c7);
        vst1q_u16(&polynomial[addr + 8*SB0], r8);
    }
    for (uint16_t addr = SB0_RES/2; addr < SB0_RES; addr+= 8) { // 9 = 288/16, inst lines: almost 120
        r0 = vld1q_u16(&w0_mem[addr]); // C(0) = A(0)*B(0)
        r1 = vld1q_u16(&w1_mem[addr]);
        r2 = vld1q_u16(&w2_mem[addr]);
        r3 = vld1q_u16(&w3_mem[addr]);
        r4 = vld1q_u16(&w4_mem[addr]);
        r5 = vld1q_u16(&w5_mem[addr]);
        r6 = vld1q_u16(&w6_mem[addr]);
        r7 = vld1q_u16(&w7_mem[addr]);
        r8 = vld1q_u16(&w8_mem[addr]); // C(f) = A(f)*B(f)


        // deal w/ theta-6
        t0 = vaddq_u16(r1, r2);      // t0 = r1         + r2
        t0 = vmulq_n_u16(t0, inv9);  // t0 = (1/9)*r1   + (1/9)*r2 --->
        t0 = vshlq_n_u16(t0, 1);     // t0 = (2/9)*r1     + (2/9)*r2
        t1 = vaddq_u16(r3, r4);      // t1 = r3         + r4
        t1 = vmulq_n_u16(t1, inv45); // t1 = (1/45)*r3  + (1/45)*r4
        t2 = vaddq_u16(r6, r7);      // t2 = r6         + r7
        t2 = vmulq_n_u16(t2, inv45); // t2 = (1/45)*r6  + (1/45)*r7
        t3 = vmulq_n_u16(r8, 21);    // t3 = (21)*r8
        c6 = vshrq_n_u16(t2, 1);     // t4 = (1/90)*r6 + (1/90)*r7
        c6 = vsubq_u16(c6, t3);      // c6 = (1/90)*r6 + (1/90)*r7 - (21)*r8
        c6 = vshrq_n_u16(c6, 1);     // c6 = (1/180)*r6 + (1/180)*r7 - (21/2)*r8
        c6 = vaddq_u16(c6, t1);      // c6 = (1/45)*r3  + (1/45)*r4 + (1/180)*r6 + (1/180)*r7 - (21/2)*r8
        c6 = vshrq_n_u16(c6, 1);     // c6 = (1/90)*r3  + (1/90)*r4 + (1/360)*r6 + (1/360)*r7 - (21/4)*r8
        c6 = vsubq_u16(c6, t0);
        c6 = vsubq_u16(c6, r0);      // c6 = (-1)*r0    + (-2/9)*r1 + (-2/9)*r2  + (1/90)*r3  + (1/90)*r4 + (1/360)*r6 + (1/360)*r7 - (21/4)*r8 ---> c6 fin


        // deal w/ theta-2
        t4 = vmulq_n_u16(r0, 21);  // t4 = (21)*r0
        c2 = vshrq_n_u16(t1, 1);   // c2 = (1/90)*r3 + (1/90)*r4
        c2 = vsubq_u16(c2, t4);    // c2 = -(21)*r0 + (1/90)*r3 + (1/90)*r4
        c2 = vshrq_n_u16(c2, 1);   // c2 = -(21/2)*r0 + (1/180)*r3 + (1/180)*r4
        c2 = vaddq_u16(c2, t2);    // c2 = -(21/2)*r0 + (1/180)*r3 + (1/180)*r4 + (1/45)*r6  + (1/45)*r7
        c2 = vshrq_n_u16(c2, 1);   // c2 = -(21/4)*r0 + (1/360)*r3 + (1/360)*r4 + (1/90)*r6  + (1/90)*r7
        c2 = vsubq_u16(c2, t0);
        c2 = vsubq_u16(c2, r8);    // c2 = (-21/4)*r0 + (-2/9)*r1  + (-2/9)*r2  + (1/360)*r3  + (1/360)*r4 + (1/90)*r6 + (1/90)*r7 + (-1)*r8 ---> c2 fin


        // deal w/ theta-4
        t2 = vaddq_u16(r6, r7);     // t2 = r6 + r7, note: cannot inherit from (1/90), 3 bit loss already
        t2 = vaddq_u16(t2, r3);     // t2 = r3 + r6 + r7
        t2 = vaddq_u16(t2, r4);     // t2 = r3         + r4        + r6         + r7
        t2 = vmulq_n_u16(t2, inv9); // t2 = (1/9)*r3   + (1/9)*r4  + (1/9)*r6   + (1/9)*r7
        t0 = vmulq_n_u16(t0, 17);   // t0 = (34/9)*r1  + (34/9)*r2
        c4 = vshrq_n_u16(t2, 1);    // c4 = (1/18)*r3   + (1/18)*r4  + (1/18)*r6   + (1/18)*r7
        c4 = vsubq_u16(t0, c4);
        c4 = vaddq_u16(c4, t4);
        c4 = vaddq_u16(c4, t3);     // c4 = (21)*r0 + (34/9)*r1  + (34/9)*r2 - (1/18)*r3   - (1/18)*r4  - (1/18)*r6   - (1/18)*r7 + (21)*r8
        c4 = vshrq_n_u16(c4, 2);    // c4 = (21/4)*r0  + (17/18)*r1  + (17/18)*r2 - (1/72)*r3   - (1/72)*r4  - (1/72)*r6   - (1/72)*r7 + (21/2)*r8 ---> c4 fin


        // deal w/ theta-7 and theta-1
        t0 = vmulq_n_u16(r3, inv45);  // t0 = (1/15)*r3
        t1 = vmulq_n_u16(r4, inv225); // t1 = (1/75)*r4
        t4 = vaddq_u16(t0, t1);       // t4 = (1/15)*r3 + (1/75)*r4
        t0 = vsubq_u16(t0, t1);
        t0 = vmulq_n_u16(t0, 3);      // t0 = (1/15)*r3 + (-1/75)*r4
        c1 = vshrq_n_u16(t0, 1);      // c1 = (1/30)*r3 + (-1/150)*r4
        t0 = vmulq_n_u16(r6, inv225); // t0 = (1/225)*r6
        t1 = vmulq_n_u16(r7, inv315); // t1 = (1/315)*r7
        t2 = vsubq_u16(t0, t1);
        t2 = vmulq_n_u16(t2, 3);      // t2 = (1/75)*r6 - (1/105)*r7
        c7 = vaddq_u16(t0, t1);       // c7 = (1/225)*r6 + (1/315)*r7
        c7 = vshrq_n_u16(c7, 1);      // c7 = (1/450)*r6 + (1/630)*r7
        t0 = vmulq_n_u16(r2, inv3);   // t0 = (1/3)*r2
        t1 = vmulq_n_u16(r5, inv525); // t1 = (1/525)*r5
        c1 = vaddq_u16(c1, t0);
        c1 = vsubq_u16(c1, t1);
        c1 = vaddq_u16(c1, t2);       // c1 = (1/3)*r2 + (1/30)*r3 + (-1/150)*r4 - (1/525)*r5 + (1/75)*r6 - (1/105)*r7
        c1 = vshrq_n_u16(c1, 1);      // c1 = (1/6)*r2 + (1/60)*r3 + (-1/300)*r4 - (1/1050)*r5 + (1/150)*r6 - (1/210)*r7
        t0 = vmulq_n_u16(r2, inv9);   // t0 = (1/9)*r2
        c7 = vsubq_u16(t1, c7);       // c7 = (1/525)*r5 - (1/450)*r6 - (1/630)*r7
        c7 = vaddq_u16(c7, t0);
        c7 = vsubq_u16(c7, t4);       // c7 = (1/9)*r2 - (1/45)*r3 - (1/225)*r4 + (1/525)*r5 - (1/450)*r6 - (1/630)*r7
        c7 = vshrq_n_u16(c7, 1);      // c7 = (1/18)*r2 - (1/90)*r3 - (1/450)*r4 + (1/1050)*r5 - (1/900)*r6 - (1/1260)*r7
        t0 = vmulq_n_u16(r0, inv3);   // t0 = (1/3)*r0
        t1 = vmulq_n_u16(r8, 3);      // t1 = 3*r8
        t0 = vsubq_u16(t0, t1);       // t0 = (1/3)*r0 - 3*r8
        t1 = vmulq_n_u16(r1, inv3);   // t1 = (1/3)*r1
        t2 = vmulq_n_u16(r1, inv9);   // t2 = (1/9)*r1
        c1 = vsubq_u16(c1, t0);
        c1 = vsubq_u16(c1, t1);       // c1 = -(1/3)*r0 - (1/3)*r1 + (1/6)*r2 + (1/60)*r3 + (-1/300)*r4 - (1/1050)*r5 + (1/150)*r6 - (1/210)*r7 + 3*r8 ---> c1 fin
        c7 = vaddq_u16(c7, t0);
        c7 = vaddq_u16(c7, t2);       // c7 = (1/3)*r0 + (1/9)*r1 + (1/18)*r2 - (1/90)*r3 - (1/450)*r4 + (1/1050)*r5 - (1/900)*r6 - (1/1260)*r7 - 3*r8 ---> c7 fin


        // deal w/ theta-5 and theta-3
        t3 = vmulq_n_u16(r5, inv25);                 // t3 = (1/25)*r5
        t5 = vmulq_n_u16(r7, inv45);                 // t5 = (1/45)*r7
        t0 = vmulq_n_u16(r2, (uint16_t)(47*inv9));   // t0 = (47/9)*r2
        t1 = vmulq_n_u16(r3, (uint16_t)(31*inv45));  // t1 = (31/45)*r3
        t2 = vmulq_n_u16(r4, (uint16_t)(29*inv225)); //t2 = (29/225)*r4
        t4 = vmulq_n_u16(r6, (uint16_t)(23*inv225)); //t4 = (23/225)*r6
        c3 = vaddq_u16(t0, t1);                      // c3 = (47/9)*r 2 + (31/45)*r3
        c3 = vsubq_u16(t2, c3);
        c3 = vaddq_u16(c3, t3);
        c3 = vsubq_u16(c3, t4);
        c3 = vaddq_u16(c3, t5);                      // c3 = -(47/9)*r2 - (31/45)*r3 + (29/225)*r4 + (1/25)*r5 - (23/225)*r6 + (1/45)*r7
        c3 = vshrq_n_u16(c3, 1);                     // c3 = -(47/18)*r2 - (31/90)*r3 + (29/450)*r4 + (1/50)*r5 - (23/450)*r6 + (1/90)*r7
        t0 = vmulq_n_u16(r2, (uint16_t)(5*inv9));    // t0 = (5/9)*r2
        t1 = vmulq_n_u16(r3, (uint16_t)(29*inv45));  // t1 = (29/45)*r3
        t2 = vmulq_n_u16(r4, (uint16_t)(19*inv225)); // t2 = (19/225)*r4
        t4 = vmulq_n_u16(r6, (uint16_t)(13*inv225)); // t4 = (13/225)*r6
        c5 = vsubq_u16(t1, t0);                      // c5 = -(5/9)*r2 + (29/45)*r3
        c5 = vsubq_u16(c5, t2);
        c5 = vsubq_u16(c5, t3);
        c5 = vaddq_u16(c5, t4);
        c5 = vaddq_u16(c5, t5);                      // c5 = -(5/9)*r2 + (29/45)*r3 - (19/225)*r4 - (1/25)*r5 + (13/225)*r6 + (1/45)*r7
        c5 = vshrq_n_u16(c5, 1);                     // c5 = -(5/18)*r2 + (29/90)*r3 - (19/450)*r4 - (1/50)*r5 + (13/450)*r6 + (1/90)*r7
        t0 = vmulq_n_u16(r0, 7);                     // t0 = 7*r0
        t3 = vmulq_n_u16(r1, (uint16_t)(29*inv9));   // t3 = (29/9)*r1
        t1 = vmulq_n_u16(r1, (uint16_t)(55*inv9));   // t1 = (55/9)*r1
        t2 = vmulq_n_u16(r8, 63);                    // t2 = 63*r8
        t0 = vsubq_u16(t0, t2);                      // t0 = 7*r0 - 63*r8
        c3 = vaddq_u16(c3, t0);
        c3 = vaddq_u16(c3, t1);                      // c3 = 7*r0 + (55/9)*r1 - (47/18)*r2 - (31/90)*r3 + (29/450)*r4 + (1/50)*r5 - (23/450)*r6 + (1/90)*r7 - 63*r8
        c3 = vshrq_n_u16(c3, 2);                     // c3 = (7/4)*r0 + (55/36)*r1 - (47/72)*r2 - (31/360)*r3 + (29/1800)*r4 + (1/200)*r5 - (23/1800)*r6 + (1/360)*r7 - (63/4)*r8 ---> c3 fin
        c5 = vsubq_u16(c5, t0);
        c5 = vsubq_u16(c5, t3);                      // c5 = -7*r0 - (29/9)*r1 - (5/18)*r2 + (29/90)*r3 - (19/450)*r4 - (1/50)*r5 + (13/450)*r6 + (1/90)*r7 + 63*r8
        c5 = vshrq_n_u16(c5, 2);                     // c5 = -(7/4)*r0 - (29/36)*r1 - (5/72)*r2 + (29/360)*r3 - (19/1800)*r4 - (1/200)*r5 + (13/1800)*r6 + (1/360)*r7 + (63/4)*r8 ---> c5 fin

        t0 = vld1q_u16(&polynomial[addr + 0*SB0]);
        r0 = vaddq_u16(t0, r0);
        vst1q_u16(&polynomial[addr + 0*SB0], r0);

        t0 = vld1q_u16(&polynomial[addr + 1*SB0]);
        c1 = vaddq_u16(t0, c1);
        vst1q_u16(&polynomial[addr + 1*SB0], c1);

        t0 = vld1q_u16(&polynomial[addr + 2*SB0]);
        c2 = vaddq_u16(t0, c2);
        vst1q_u16(&polynomial[addr + 2*SB0], c2);

        t0 = vld1q_u16(&polynomial[addr + 3*SB0]);
        c3 = vaddq_u16(t0, c3);
        vst1q_u16(&polynomial[addr + 3*SB0], c3);

        t0 = vld1q_u16(&polynomial[addr + 4*SB0]);
        c4 = vaddq_u16(t0, c4);
        vst1q_u16(&polynomial[addr + 4*SB0], c4);

        t0 = vld1q_u16(&polynomial[addr + 5*SB0]);
        c5 = vaddq_u16(t0, c5);
        vst1q_u16(&polynomial[addr + 5*SB0], c5);

        t0 = vld1q_u16(&polynomial[addr + 6*SB0]);
        c6 = vaddq_u16(t0, c6);
        vst1q_u16(&polynomial[addr + 6*SB0], c6);

        t0 = vld1q_u16(&polynomial[addr + 7*SB0]);
        c7 = vaddq_u16(t0, c7);
        vst1q_u16(&polynomial[addr + 7*SB0], c7);

        vst1q_u16(&polynomial[addr + 8*SB0], r8);
    }
}



static void poly_neon_reduction(uint16_t *polynomial, uint16_t *tmp) {
    uint16x8_t mask;
    uint16x8x3_t res, tmp1, tmp2;
    mask = vdupq_n_u16(MASK);
    for (uint16_t addr = 0; addr < 680; addr += 24) {
        tmp2 = vld1q_u16_x3(&tmp[addr]);
        tmp1 = vld1q_u16_x3(&tmp[addr + NTRU_N]);
        res.val[0] = vaddq_u16(tmp1.val[0], tmp2.val[0]);
        res.val[1] = vaddq_u16(tmp1.val[1], tmp2.val[1]);
        res.val[2] = vaddq_u16(tmp1.val[2], tmp2.val[2]);
        res.val[0] = vandq_u16(res.val[0], mask);
        res.val[1] = vandq_u16(res.val[1], mask);
        res.val[2] = vandq_u16(res.val[2], mask);
        vst1q_u16_x3(&polynomial[addr], res);
    }
}

void tc33_mul(uint16_t *restrict polyC[9], uint16_t *restrict polyA[9], uint16_t *restrict polyB[9]) {
    uint16_t tmp_aabb[9*SB2*50+224*8*2], tmp_cc[9*SB2_RES*25+224*32]; // 50, 25
    uint16_t *tmp_aa = &tmp_aabb[9*SB2*0],
             *tmp_bb = &tmp_aabb[9*SB2*25+224*8],
             *tmp_aa1 = &tmp_aabb[9*SB2*25],
             *tmp_bb1 = &tmp_aabb[9*SB2*50+224*8],
             *tmp_cc1 = &tmp_cc[9*SB2_RES*25];

    tc33(&tmp_aa[0*25*SB2], polyA[0]); /* 3.1k cycles, 3.1k/18 = 0.2k each */
    tc33(&tmp_bb[0*25*SB2], polyB[0]);

    tc33(&tmp_aa[1*25*SB2], polyA[1]);
    tc33(&tmp_bb[1*25*SB2], polyB[1]);

    tc33(&tmp_aa[2*25*SB2], polyA[2]);
    tc33(&tmp_bb[2*25*SB2], polyB[2]);

    tc33(&tmp_aa[3*25*SB2], polyA[3]);
    tc33(&tmp_bb[3*25*SB2], polyB[3]);

    tc33(&tmp_aa[4*25*SB2], polyA[4]);
    tc33(&tmp_bb[4*25*SB2], polyB[4]);

    tc33(&tmp_aa[5*25*SB2], polyA[5]);
    tc33(&tmp_bb[5*25*SB2], polyB[5]);

    tc33(&tmp_aa[6*25*SB2], polyA[6]);
    tc33(&tmp_bb[6*25*SB2], polyB[6]);

    tc33(&tmp_aa[7*25*SB2], polyA[7]);
    tc33(&tmp_bb[7*25*SB2], polyB[7]);

    tc33(&tmp_aa[8*25*SB2], polyA[8]);
    tc33(&tmp_bb[8*25*SB2], polyB[8]);

/* Split 225 16x16 into 1 and 224 */

//1x 16x16
    schoolbook_16x16(&tmp_cc[0], &tmp_aa[0] , &tmp_bb[0]); /* without this function the whole code runs 0.5k cycles slower */

//224x 16x16
    k2(&tmp_aa1[0], &tmp_aa[16]); /* 1.5k cycles, 1.5k/2 = 0.7k each */
    k2(&tmp_bb1[0], &tmp_bb[16]);
    schoolbook_8x8(&tmp_cc[32], &tmp_aa[16], &tmp_bb[16]); /* 18k cycles */
    ik2(&tmp_cc[32], &tmp_cc1[0]); /* 2.4k cycles */


    itc33(polyC[0], &tmp_cc[0*25*SB2_RES]); /* 4.2k cycles, 4.2k/9 = 0.5k each */
    itc33(polyC[1], &tmp_cc[1*25*SB2_RES]);
    itc33(polyC[2], &tmp_cc[2*25*SB2_RES]);
    itc33(polyC[3], &tmp_cc[3*25*SB2_RES]);
    itc33(polyC[4], &tmp_cc[4*25*SB2_RES]);
    itc33(polyC[5], &tmp_cc[5*25*SB2_RES]);
    itc33(polyC[6], &tmp_cc[6*25*SB2_RES]);
    itc33(polyC[7], &tmp_cc[7*25*SB2_RES]);
    itc33(polyC[8], &tmp_cc[8*25*SB2_RES]);
}

void poly_mul_neon(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB) {
    uint16_t *kaw[9], *kbw[9], *kcw[9];
    uint16_t tmp_ab[SB0 * 9 * 2];
    uint16_t tmp_c[SB0_RES * 9];

    kaw[0] = &tmp_ab[0 * SB0]; // A(0)
    kbw[0] = &tmp_ab[1 * SB0]; // B(0)

    kaw[1] = &tmp_ab[2 * SB0];
    kbw[1] = &tmp_ab[3 * SB0];

    kaw[2] = &tmp_ab[4 * SB0];
    kbw[2] = &tmp_ab[5 * SB0];

    kaw[3] = &tmp_ab[6 * SB0];
    kbw[3] = &tmp_ab[7 * SB0];

    kaw[4] = &tmp_ab[8 * SB0];
    kbw[4] = &tmp_ab[9 * SB0];

    kaw[5] = &tmp_ab[10 * SB0];
    kbw[5] = &tmp_ab[11 * SB0];

    kaw[6] = &tmp_ab[12 * SB0];
    kbw[6] = &tmp_ab[13 * SB0];

    kaw[7] = &tmp_ab[14 * SB0];
    kbw[7] = &tmp_ab[15 * SB0];

    kaw[8] = &tmp_ab[16 * SB0]; // A(f)
    kbw[8] = &tmp_ab[17 * SB0]; // B(f)

    // kcw
    kcw[0] = &tmp_c[0 * SB0_RES];
    kcw[1] = &tmp_c[1 * SB0_RES];
    kcw[2] = &tmp_c[2 * SB0_RES];
    kcw[3] = &tmp_c[3 * SB0_RES];
    kcw[4] = &tmp_c[4 * SB0_RES];
    kcw[5] = &tmp_c[5 * SB0_RES];
    kcw[6] = &tmp_c[6 * SB0_RES];
    kcw[7] = &tmp_c[7 * SB0_RES];
    kcw[8] = &tmp_c[8 * SB0_RES];

    tc5(kaw, polyA); /* 0.6k cycles */
    tc5(kbw, polyB); /* 0.6k cycles */

    tc33_mul(kcw, kaw, kbw);

    itc5(tmp_ab, kcw); /* 3.3k cycles */

    // Ring reduction, Reduce from 1440 -> 720
    poly_neon_reduction(polyC, tmp_ab); /* 0.4k cycles */
}



