
#include <stdio.h>
#include <arm_neon.h>

#include "params.h"
#include "poly.h"
#include "batch_multiplication.h"

#include "tmvp.h"

static const unsigned int ZREV[4]  = {0xffffffff, 0x0100ffff, 0x05040302, 0x09080706};

/*
w points to 15 (ordered) output 8x8 Toeplitz Matrix,
src points to 1 input 48x48 Toeplitz Matrix
*/

void ittc32(uint16_t *restrict w, uint16_t *restrict src){
             uint16x8_t z0, z1, z2, z3, z4;
             uint16x8_t p0, p1, p2, p3, p4;
             uint16x8_t t0, t1;
             uint16x8_t z20, z21, z22, z23, z24;
             uint16x8_t p20, p21, p22, p23, p24;
             uint16x8_t t20, t21;
             uint16x8_t z80, z81, z82, z83, z84;
             uint16x8_t p80, p81, p82, p83, p84;
             uint16x8_t t80, t81;
             uint16x8_t z820, z821, z822, z823, z824;
             uint16x8_t p820, p821, p822, p823, p824;
             uint16x8_t t820, t821;

             for (uint16_t num = 0; num < 5; num++) {
                z0 = vld1q_u16(&src[1*SB2+num*2*SB1]);
                z1 = vld1q_u16(&src[2*SB2+num*2*SB1]);
                z2 = vld1q_u16(&src[3*SB2+num*2*SB1]);
                z3 = vld1q_u16(&src[4*SB2+num*2*SB1]);
                z4 = vld1q_u16(&src[5*SB2+num*2*SB1]);

                p3 = vsubq_u16(z3, z1); //p3 = z3 - z1
                p0 = vsubq_u16(z4, z2);
                p0 = vshlq_n_u16(p0, 1);
                p0 = vaddq_u16(p0, p3); //p0 =  2*z4 + z3 -2*z2 - z1

                t0 = vsubq_u16(z2, z3); //t0 = z2 - z3
                p1 = vshlq_n_u16(z2, 2);
                p1 = vaddq_u16(p1, z3);
                p1 = vaddq_u16(p1, z1);
                p1 = vsubq_u16(p1, t0); //p1 = 2*z3 + 3*z2 + z1

                p2 = vsubq_u16(t0, p3); //p2 = -2*z3 + z2 + z1

                p4 = vsubq_u16(z0, z2);
                t1 = vshlq_n_u16(p3, 1);
                p4 = vsubq_u16(p4, t1); //p4 = -2*z3 - z2 + 2*z1 + *z0



                z20 = vld1q_u16(&src[0*SB2+num*2*SB1]);
                z21 = z0;
                z22 = z1;
                z23 = z2;
                z24 = z3;

                p23 = vsubq_u16(z23, z21); //p23 = z23 - z21
                p20 = vsubq_u16(z24, z22);
                p20 = vshlq_n_u16(p20, 1);
                p20 = vaddq_u16(p20, p23); //p20 =  2*z24 + z23 -2*z22 - z21

                t20 = vsubq_u16(z22, z23); //t20 = z22 - z23
                p21 = vshlq_n_u16(z22, 2);
                p21 = vaddq_u16(p21, z23);
                p21 = vaddq_u16(p21, z21);
                p21 = vsubq_u16(p21, t20); //p21 = 2*z23 + 3*z22 + z21

                p22 = vsubq_u16(t20, p23); //p22 = -2*z23 + z22 + z21

                p24 = vsubq_u16(z20, z22);
                t21 = vshlq_n_u16(p23, 1);
                p24 = vsubq_u16(p24, t21); //p24 = -2*z23 - z22 + 2*z21 + z20




                z80 = vld1q_u16(&src[1*SB2+8+num*2*SB1]);
                z81 = vld1q_u16(&src[2*SB2+8+num*2*SB1]);
                z82 = vld1q_u16(&src[3*SB2+8+num*2*SB1]);
                z83 = vld1q_u16(&src[4*SB2+8+num*2*SB1]);
                z84 = vld1q_u16(&src[5*SB2+8+num*2*SB1]);

                p83 = vsubq_u16(z83, z81); //p83 = z83 - z81
                p80 = vsubq_u16(z84, z82);
                p80 = vshlq_n_u16(p80, 1);
                p80 = vaddq_u16(p80, p83); //p80 =  2*z84 + z83 -2*z82 - z81

                t80 = vsubq_u16(z82, z83); //t80 = z82 - z83
                p81 = vshlq_n_u16(z82, 2);
                p81 = vaddq_u16(p81, z83);
                p81 = vaddq_u16(p81, z81);
                p81 = vsubq_u16(p81, t80); //p81 = 2*z83 + 3*z82 + z81

                p82 = vsubq_u16(t80, p83); //p82 = -2*z83 + z82 + z81

                p84 = vsubq_u16(z80, z82);
                t81 = vshlq_n_u16(p83, 1);
                p84 = vsubq_u16(p84, t81); //p84 = -2*z83 - z82 + 2*z81 + z80




                z820 = vld1q_u16(&src[0*SB2+8+num*2*SB1]);
                z821 = z80;
                z822 = z81;
                z823 = z82;
                z824 = z83;

                p823 = vsubq_u16(z823, z821); //p823 = z823 - z821
                p820 = vsubq_u16(z824, z822);
                p820 = vshlq_n_u16(p820, 1);
                p820 = vaddq_u16(p820, p823); //p820 =  2*z824 + z823 -2*z822 - z821

                t820 = vsubq_u16(z822, z823); //t820 = z822 - z823
                p821 = vshlq_n_u16(z822, 2);
                p821 = vaddq_u16(p821, z823);
                p821 = vaddq_u16(p821, z821);
                p821 = vsubq_u16(p821, t820); //p821 = 2*z823 + 3*z822 + z821

                p822 = vsubq_u16(t820, p823); //p822 = -2*z823 + z822 + z821

                p824 = vsubq_u16(z820, z822);
                t821 = vshlq_n_u16(p823, 1);
                p824 = vsubq_u16(p824, t821); //p824 = -2*z823 - z822 + 2*z821 + z820



                uint16x8_t a0, a1, a2;
                uint16x8_t r0, r1;
                //p0
        a1 = p0;
        a2 = p80;
        a0 = p820;


        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+8 ], r0);
        vst1q_u16(&w[num*15*SB2+24], r1);
        vst1q_u16(&w[num*15*SB2+40], a1);



        a1 = p820;
        a0 = p20;
        a2 = p0;

        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+0 ], r0);
        vst1q_u16(&w[num*15*SB2+16], r1);
        vst1q_u16(&w[num*15*SB2+32], a1);
                //p1
        a1 = p1;
        a2 = p81;
        a0 = p821;


        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+8 +1*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+24+1*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+40+1*3*SB2], a1);



        a1 = p821;
        a0 = p21;
        a2 = p1;

        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+0 +1*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+16+1*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+32+1*3*SB2], a1);
                //p2
        a1 = p2;
        a2 = p82;
        a0 = p822;


        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+8 +2*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+24+2*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+40+2*3*SB2], a1);



        a1 = p822;
        a0 = p22;
        a2 = p2;

        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+0 +2*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+16+2*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+32+2*3*SB2], a1);
                //p3
        a1 = p3;
        a2 = p83;
        a0 = p823;


        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+8 +3*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+24+3*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+40+3*3*SB2], a1);



        a1 = p823;
        a0 = p23;
        a2 = p3;

        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+0 +3*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+16+3*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+32+3*3*SB2], a1);
                //p4
        a1 = p4;
        a2 = p84;
        a0 = p824;


        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+8 +4*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+24+4*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+40+4*3*SB2], a1);



        a1 = p824;
        a0 = p24;
        a2 = p4;

        r0 = vsubq_u16(a1, a2);
        r1 = vsubq_u16(a0, a1);

        vst1q_u16(&w[num*15*SB2+0 +4*3*SB2], r0);
        vst1q_u16(&w[num*15*SB2+16+4*3*SB2], r1);
        vst1q_u16(&w[num*15*SB2+32+4*3*SB2], a1);
             }
}
/*
w points to 5 (ordered) output 48x48 Toeplitz Matrix,
src points to 1 input 144x144 Toeplitz Matrix
*/
void ittc3(uint16_t *restrict w, uint16_t *restrict src){
    uint16_t *w0_mem =  &w[1*SB1],
             *w1_mem =  &w[3*SB1],
             *w2_mem =  &w[5*SB1],
             *w3_mem =  &w[7*SB1],
             *w4_mem =  &w[9*SB1],
             *w20_mem = &w[0*SB1],
             *w21_mem = &w[2*SB1],
             *w22_mem = &w[4*SB1],
             *w23_mem = &w[6*SB1],
             *w24_mem = &w[8*SB1],

             *c2 = &src[0*SB1+3*SB1],
             *c3 = &src[1*SB1+3*SB1],
             *c4 = &src[2*SB1+3*SB1],
             *c1 = &src[2*SB1],
             *c0 = &src[1*SB1],
             *c20= &src[0*SB1];

             uint16x8_t z0, z1, z2, z3, z4;
             uint16x8_t p0, p1, p2, p3, p4;
             uint16x8_t t0, t1;
             uint16x8_t z20, z21, z22, z23, z24;
             uint16x8_t p20, p21, p22, p23, p24;
             uint16x8_t t20, t21;

             for (uint16_t addr = 0; addr < SB1; addr+= 8) {
                z0 = vld1q_u16(&c0[addr]);
                z1 = vld1q_u16(&c1[addr]);
                z2 = vld1q_u16(&c2[addr]);
                z3 = vld1q_u16(&c3[addr]);
                z4 = vld1q_u16(&c4[addr]);

                p3 = vsubq_u16(z3, z1); //p3 = z3 - z1
                p0 = vsubq_u16(z4, z2);
                p0 = vshlq_n_u16(p0, 1);
                p0 = vaddq_u16(p0, p3); //p0 =  2*z4 + z3 -2*z2 - z1

                t0 = vsubq_u16(z2, z3); //t0 = z2 - z3
                p1 = vshlq_n_u16(z2, 2);
                p1 = vaddq_u16(p1, z3);
                p1 = vaddq_u16(p1, z1);
                p1 = vsubq_u16(p1, t0); //p1 = 2*z3 + 3*z2 + z1

                p2 = vsubq_u16(t0, p3); //p2 = -2*z3 + z2 + z1

                p4 = vsubq_u16(z0, z2);
                t1 = vshlq_n_u16(p3, 1);
                p4 = vsubq_u16(p4, t1); //p4 = -2*z3 - z2 + 2*z1 + z0

                vst1q_u16(&w0_mem[addr], p0);
                vst1q_u16(&w1_mem[addr], p1);
                vst1q_u16(&w2_mem[addr], p2);
                vst1q_u16(&w3_mem[addr], p3);
                vst1q_u16(&w4_mem[addr], p4);

                z20 = vld1q_u16(&c20[addr]);
                z21 = z0;
                z22 = z1;
                z23 = z2;
                z24 = z3;

                p23 = vsubq_u16(z23, z21); //p23 = z23 - z21
                p20 = vsubq_u16(z24, z22);
                p20 = vshlq_n_u16(p20, 1);
                p20 = vaddq_u16(p20, p23); //p20 =  2*z24 + z23 -2*z22 - z21

                t20 = vsubq_u16(z22, z23); //t20 = z22 - z23
                p21 = vshlq_n_u16(z22, 2);
                p21 = vaddq_u16(p21, z23);
                p21 = vaddq_u16(p21, z21);
                p21 = vsubq_u16(p21, t20); //p21 = 2*z23 + 3*z22 + z21

                p22 = vsubq_u16(t20, p23); //p22 = -2*z23 + z22 + z21

                p24 = vsubq_u16(z20, z22);
                t21 = vshlq_n_u16(p23, 1);
                p24 = vsubq_u16(p24, t21); //p24 = -2*z23 - z22 + 2*z21 + z20

                vst1q_u16(&w20_mem[addr], p20);
                vst1q_u16(&w21_mem[addr], p21);
                vst1q_u16(&w22_mem[addr], p22);
                vst1q_u16(&w23_mem[addr], p23);
                vst1q_u16(&w24_mem[addr], p24);
             }
}

/*
Because the input has only degree 677, omit the calculation of 680-720

w points to 9 (ordered) output 144x144 Toeplitz matrix,
polynomial points to 1 input 720x720 Toeplitz matrix
*/
void ittc5(uint16_t *restrict w, uint16_t *restrict polynomial){
    uint16_t *w0_mem = &w[1*SB0],
             *w1_mem = &w[3*SB0],
             *w2_mem = &w[5*SB0],
             *w3_mem = &w[7*SB0],
             *w4_mem = &w[9*SB0],
             *w5_mem = &w[11*SB0],
             *w6_mem = &w[13*SB0],
             *w7_mem = &w[15*SB0],
             *w8_mem = &w[17*SB0],
             *w20_mem = &w[0*SB0],
             *w21_mem = &w[2*SB0],
             *w22_mem = &w[4*SB0],
             *w23_mem = &w[6*SB0],
             *w24_mem = &w[8*SB0],
             *w25_mem = &w[10*SB0],
             *w26_mem = &w[12*SB0],
             *w27_mem = &w[14*SB0],
             *w28_mem = &w[16*SB0],

             *c4 = &polynomial[0*SB0],
             *c5 = &polynomial[1*SB0],
             *c6 = &polynomial[2*SB0],
             *c7 = &polynomial[3*SB0],
             *c8 = &polynomial[4*SB0];

             uint16x8_t z0, z1, z2, z3, z4, z5, z6, z7, z8;
             uint16x8_t Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7;
             uint16x8_t p0, p1, p2, p3, p4, p5, p6, p7, p8;
             uint16x8_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
             uint16x8_t tmp;


            for (uint16_t addr = 0; addr < 8*5; addr+= 8) {
                z0 = vld1q_u16(&polynomial[677-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[677-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[677-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[677-SB0+addr]);
                z4 = vld1q_u16(&c4[addr]);
                z5 = vld1q_u16(&c5[addr]);
                z6 = vld1q_u16(&c6[addr]);
                z7 = vld1q_u16(&c7[addr]);
                z8 = vld1q_u16(&c8[addr]);

                //cache
                Z0 = z0;
                Z1 = z1;
                Z2 = z2;
                Z3 = z3;
                Z4 = z4;
                Z5 = z5;
                Z6 = z6;
                Z7 = z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w3_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w4_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w7_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w6_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w0_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w8_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w5_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w2_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w1_mem[addr], p1);



                z1 = Z0;
                z2 = Z1;
                z3 = Z2;
                z4 = Z3;
                z5 = Z4;
                z6 = Z5;
                z7 = Z6;
                z8 = Z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w23_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w24_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w27_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w26_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w20_mem[addr], p0);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w25_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w22_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w21_mem[addr], p1);
             }
             for (uint16_t addr = 8*5; addr < 8*6; addr+= 8) {
                z0 = vld1q_u16(&polynomial[677-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[677-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[677-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[677-SB0+addr]);
                z4 = vld1q_u16(&c4[addr]);
                z5 = vld1q_u16(&c5[addr]);
                z6 = vld1q_u16(&c6[addr]);
                z7 = vld1q_u16(&c7[addr]);
                z8 = vld1q_u16(&c8[addr]);

                //cache
                Z0 = z0;
                Z1 = z1;
                Z2 = z2;
                Z3 = z3;
                Z4 = z4;
                Z5 = z5;
                Z6 = z6;
                Z7 = z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w3_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w4_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w7_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w6_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w0_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w8_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w5_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w2_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w1_mem[addr], p1);




                uint8x16_t zrev;
                zrev = vld1q_u8((uint8_t*)ZREV);
                z0 = vld1q_u16(&polynomial[0]);
                z0 = vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(z0), zrev));
                z1 = Z0;
                z2 = Z1;
                z3 = Z2;
                z4 = Z3;
                z5 = Z4;
                z6 = Z5;
                z7 = Z6;
                z8 = Z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w23_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w24_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w27_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w26_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w20_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w28_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w25_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w22_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w21_mem[addr], p1);
             }
             for (uint16_t addr = 8*6; addr < 8*13; addr+= 8) {
                z0 = vld1q_u16(&polynomial[677-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[677-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[677-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[677-SB0+addr]);
                z4 = vld1q_u16(&c4[addr]);
                z5 = vld1q_u16(&c5[addr]);
                z6 = vld1q_u16(&c6[addr]);
                z7 = vld1q_u16(&c7[addr]);
                z8 = vld1q_u16(&c8[addr]);

                //cache
                Z0 = z0;
                Z1 = z1;
                Z2 = z2;
                Z3 = z3;
                Z4 = z4;
                Z5 = z5;
                Z6 = z6;
                Z7 = z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w3_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w4_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w7_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w6_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w0_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w8_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w5_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w2_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w1_mem[addr], p1);




                z0 = vld1q_u16(&polynomial[677-SB0*5+addr]);
                z1 = Z0;
                z2 = Z1;
                z3 = Z2;
                z4 = Z3;
                z5 = Z4;
                z6 = Z5;
                z7 = Z6;
                z8 = Z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w23_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w24_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w27_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w26_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w20_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w28_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w25_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w22_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w21_mem[addr], p1);
             }
             for (uint16_t addr = 8*13; addr < SB0; addr+= 8) {
                z0 = vld1q_u16(&polynomial[677-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[677-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[677-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[677-SB0+addr]);
                z4 = vld1q_u16(&c4[addr]);
                z5 = vld1q_u16(&c5[addr]);
                z6 = vld1q_u16(&c6[addr]);
                z7 = vld1q_u16(&c7[addr]);

                //cache
                Z0 = z0;
                Z1 = z1;
                Z2 = z2;
                Z3 = z3;
                Z4 = z4;
                Z5 = z5;
                Z6 = z6;
                Z7 = z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w3_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w4_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w7_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w6_mem[addr], p6);



                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w8_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w5_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w2_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w1_mem[addr], p1);




                z0 = vld1q_u16(&polynomial[677-SB0*5+addr]);
                z1 = Z0;
                z2 = Z1;
                z3 = Z2;
                z4 = Z3;
                z5 = Z4;
                z6 = Z5;
                z7 = Z6;
                z8 = Z7;

                t0 = vsubq_u16(z1, z7); //t0 = z1 - z7
                t1 = vsubq_u16(z5, z3); //t1 = z5 - z3
                p5 = vshlq_n_u16(t0, 2); //p5 = 4*z1 - 4*z7
                t2 = vshlq_n_u16(t0, 3); //t2 = 8*z1 - 8*z7
                t3 = vshlq_n_u16(t1, 6);
                t3 = vsubq_u16(t3, t1); //t3 = 63*z5 -63*z3
                t4 = vmulq_n_u16(z5, 47);
                t4 = vmlaq_n_u16(t4, z3, 5);//t4 = 47*z5 + 5*z3
                t5 = vshlq_n_u16(z7, 4); //t5 = 16*z7
                t6 = vshlq_n_u16(z4, 2);
                t7 = vshlq_n_u16(z4, 6);
                t7 = vaddq_u16(t7, t6); //t7 = 68*z4
                t6 = vaddq_u16(t6, z4); //t6 = 5*z4
                t8 = vmulq_n_u16(z1, (uint16_t)(-2*inv5)); //t8 = -2/5*z1
                t9 = vmulq_n_u16(z7, (uint16_t)(2*45*inv75)); //t9 = 45*2/75*z7
                p7 = vaddq_u16(z5, z3); //p7 = z5 + z3
                p2 = vsubq_u16(p5, t4);


                p4 = vsubq_u16(z6, t6);
                tmp = vshlq_n_u16(z2, 2);   //tmp = 4*z2
                p4 = vaddq_u16(p4, tmp); //p4 = z6 - 5*z4 + 4*z2

                p6 = vsubq_u16(z2, t6);
                tmp = vshlq_n_u16(z6, 2);   //tmp = 4*z6
                p6 = vaddq_u16(p6, tmp); //p6 = 4*z6 - 5*z4 + z2


                p3 = vshlq_n_u16(t1, 5);
                p3 = vsubq_u16(p7, p3);
                tmp = vshlq_n_u16(z3, 2); //tmp = 4*z3
                p3 = vsubq_u16(p3, tmp);
                p3 = vaddq_u16(p3, p4);
                p3 = vsubq_u16(p3, p5);
                tmp = vaddq_u16(z7, z7); //tmp = 2*z7
                p3 = vaddq_u16(p3, tmp);
                vst1q_u16(&w23_mem[addr], p3);

                p4 = vsubq_u16(p4, t9);
                tmp = vaddq_u16(t8, t8);
                p4 = vaddq_u16(p4, tmp);
                p4 = vmlaq_n_u16(p4, z5, (uint16_t)(45*29*inv225));
                p4 = vmlaq_n_u16(p4, z3, (uint16_t)(-45*19*inv225));
                vst1q_u16(&w24_mem[addr], p4);

                p7 = vaddq_u16(p7, p6);
                p7 = vmlaq_n_u16(p7, z7, (uint16_t)(-45*4*inv105));
                p7 = vmlaq_n_u16(p7, z1, (uint16_t)(-45*2*inv315));
                vst1q_u16(&w27_mem[addr], p7);

                tmp = vaddq_u16(t9, t9);
                p6 = vaddq_u16(p6, tmp);
                p6 = vaddq_u16(p6, t8);
                p6 = vmlaq_n_u16(p6, z5, (uint16_t)(-45*23*inv225));
                p6 = vmlaq_n_u16(p6, z3, (uint16_t)(45*13*inv225));
                vst1q_u16(&w26_mem[addr], p6);


                p0 = vsubq_u16(z8, z2);
                p0 = vshlq_n_u16(p0, 2);
                tmp = vsubq_u16(z4, z6);
                p0 = vmlaq_n_u16(p0, t0, (uint16_t)(4*inv3));
                p0 = vmlaq_n_u16(p0, t1, 7);
                p0 = vmlaq_n_u16(p0, tmp, 21);
                vst1q_u16(&w20_mem[addr], p0);

                p8 = vsubq_u16(z0, z6);
                p8 = vshlq_n_u16(p8, 2);
                tmp = vsubq_u16(z4, z2);
                p8 = vsubq_u16(p8, p5);
                p8 = vsubq_u16(p8, t2);
                p8 = vsubq_u16(p8, t3);
                p8 = vmlaq_n_u16(p8, tmp, 21);
                vst1q_u16(&w28_mem[addr], p8);

                p5 = vmlaq_n_u16(p5, t1, 21);
                vst1q_u16(&w25_mem[addr], p5);

                p1 = vaddq_u16(z2, z6);
                p1 = vshlq_n_u16(p1, 4);
                p1 = vsubq_u16(p1, t7);

                p2 = vsubq_u16(p2, p1);
                p2 = vaddq_u16(p2, t5);
                vst1q_u16(&w22_mem[addr], p2);

                p1 = vsubq_u16(t2, p1);
                p1 = vsubq_u16(p1, t5);
                p1 = vaddq_u16(p1, t3);
                p1 = vaddq_u16(p1, t4);
                vst1q_u16(&w21_mem[addr], p1);
             }
}

/*
Because the output has only degree 677, omit the calculation of 680-720

w points to 9 (ordered) input size-144 vectors,
polynomial points to 1 output size-720 vector
*/
void ttc5(uint16_t *restrict polynomial, uint16_t *restrict w){
    uint16_t *w0_mem = &w[0*SB0],
             *w1_mem = &w[1*SB0],
             *w2_mem = &w[2*SB0],
             *w3_mem = &w[3*SB0],
             *w4_mem = &w[4*SB0],
             *w5_mem = &w[5*SB0],
             *w6_mem = &w[6*SB0],
             *w7_mem = &w[7*SB0],
             *w8_mem = &w[8*SB0],
             *dst0 = &polynomial[0*SB0],
             *dst1 = &polynomial[1*SB0],
             *dst2 = &polynomial[2*SB0],
             *dst3 = &polynomial[3*SB0],
             *dst4 = &polynomial[4*SB0];

             uint16x8_t p0, p1, p2, p3, p4, p5, p6, p7, p8;
             uint16x8_t t0, t1, t2, t3, t4, t5;
             uint16x8_t k0, k1, k2, k3, k4;
             uint16x8_t tmp;
             uint16x8_t mask;
             mask = vdupq_n_u16(MASK);

             for (uint16_t addr = 0; addr < 8*13; addr+= 8){
                p0 = vld1q_u16(&w0_mem[addr]);
                p1 = vld1q_u16(&w1_mem[addr]);
                p1 = vmulq_n_u16(p1, inv9);
                p2 = vld1q_u16(&w2_mem[addr]);
                p2 = vmulq_n_u16(p2, inv9);
                p3 = vld1q_u16(&w3_mem[addr]);
                p3 = vmulq_n_u16(p3, inv45);
                p4 = vld1q_u16(&w4_mem[addr]);
                p4 = vmulq_n_u16(p4, inv45);
                p5 = vld1q_u16(&w5_mem[addr]);
                p5 = vmulq_n_u16(p5, inv525);
                p6 = vld1q_u16(&w6_mem[addr]);
                p6 = vmulq_n_u16(p6, inv45);
                p7 = vld1q_u16(&w7_mem[addr]);
                p7 = vmulq_n_u16(p7, inv45);
                p8 = vld1q_u16(&w8_mem[addr]);

                t0 = vaddq_u16(p1, p2); //t0 = p1 + p2
                t1 = vsubq_u16(p1, p2); //t1 = p1 - p2
                t2 = vaddq_u16(p3, p4); //t2 = p3 + p4
                t3 = vsubq_u16(p3, p4); //t3 = p3 - p4
                t4 = vaddq_u16(p6, p7); //t4 = p6 + p7
                t5 = vsubq_u16(p6, p7); //t5 = p6 - p7

                k4 = vshlq_n_u16(p0, 1);
                tmp = vaddq_u16(t0, t2);
                k4 = vaddq_u16(k4, tmp);
                k4 = vaddq_u16(k4, p5);
                tmp = vshlq_n_u16(t4, 4);
                k4 = vaddq_u16(k4, tmp);
                k4 = vshrq_n_u16(k4, 3);

                tmp = vshlq_n_u16(t3, 1);
                k3 = vaddq_u16(t1, tmp);
                k3 = vmlaq_n_u16(k3, p5, 3);
                k3 = vshrq_n_u16(k3, 3);
                k3 = vaddq_u16(k3, t5);

                tmp = vaddq_u16(t2, t4);
                tmp = vshlq_n_u16(tmp, 2);
                k2 = vaddq_u16(t0, tmp);
                k2 = vmlaq_n_u16(k2, p5, 9);
                k2 = vshrq_n_u16(k2, 3);

                tmp = vshlq_n_u16(t5, 1);
                k1 = vaddq_u16(t1, tmp);
                k1 = vmlaq_n_u16(k1, p5, 27);
                k1 = vshrq_n_u16(k1, 3);
                k1 = vaddq_u16(k1, t3);


                k0 = vshlq_n_u16(p8, 1);
                k0 = vmlaq_n_u16(k0, p5, 81);
                tmp = vshlq_n_u16(t2, 4);
                k0 = vaddq_u16(k0, tmp);
                tmp = vaddq_u16(t0, t4);
                k0 = vaddq_u16(k0, tmp);
                k0 = vshrq_n_u16(k0, 3);

                k4 = vandq_u16(k4, mask);
                k3 = vandq_u16(k3, mask);
                k2 = vandq_u16(k2, mask);
                k1 = vandq_u16(k1, mask);
                k0 = vandq_u16(k0, mask);
                vst1q_u16(&dst4[addr], k4);
                vst1q_u16(&dst3[addr], k3);
                vst1q_u16(&dst2[addr], k2);
                vst1q_u16(&dst1[addr], k1);
                vst1q_u16(&dst0[addr], k0);
             }
             for (uint16_t addr = 8*13; addr < SB0; addr+= 8){
                //p0 = vld1q_u16(&w0_mem[addr]);
                p1 = vld1q_u16(&w1_mem[addr]);
                p1 = vmulq_n_u16(p1, inv9);
                p2 = vld1q_u16(&w2_mem[addr]);
                p2 = vmulq_n_u16(p2, inv9);
                p3 = vld1q_u16(&w3_mem[addr]);
                p3 = vmulq_n_u16(p3, inv45);
                p4 = vld1q_u16(&w4_mem[addr]);
                p4 = vmulq_n_u16(p4, inv45);
                p5 = vld1q_u16(&w5_mem[addr]);
                p5 = vmulq_n_u16(p5, inv525);
                p6 = vld1q_u16(&w6_mem[addr]);
                p6 = vmulq_n_u16(p6, inv45);
                p7 = vld1q_u16(&w7_mem[addr]);
                p7 = vmulq_n_u16(p7, inv45);
                p8 = vld1q_u16(&w8_mem[addr]);

                t0 = vaddq_u16(p1, p2); //t0 = p1 + p2
                t1 = vsubq_u16(p1, p2); //t1 = p1 - p2
                t2 = vaddq_u16(p3, p4); //t2 = p3 + p4
                t3 = vsubq_u16(p3, p4); //t3 = p3 - p4
                t4 = vaddq_u16(p6, p7); //t4 = p6 + p7
                t5 = vsubq_u16(p6, p7); //t5 = p6 - p7

                tmp = vshlq_n_u16(t3, 1);
                k3 = vaddq_u16(t1, tmp);
                k3 = vmlaq_n_u16(k3, p5, 3);
                k3 = vshrq_n_u16(k3, 3);
                k3 = vaddq_u16(k3, t5);

                tmp = vaddq_u16(t2, t4);
                tmp = vshlq_n_u16(tmp, 2);
                k2 = vaddq_u16(t0, tmp);
                k2 = vmlaq_n_u16(k2, p5, 9);
                k2 = vshrq_n_u16(k2, 3);

                tmp = vshlq_n_u16(t5, 1);
                k1 = vaddq_u16(t1, tmp);
                k1 = vmlaq_n_u16(k1, p5, 27);
                k1 = vshrq_n_u16(k1, 3);
                k1 = vaddq_u16(k1, t3);


                k0 = vshlq_n_u16(p8, 1);
                k0 = vmlaq_n_u16(k0, p5, 81);
                tmp = vshlq_n_u16(t2, 4);
                k0 = vaddq_u16(k0, tmp);
                tmp = vaddq_u16(t0, t4);
                k0 = vaddq_u16(k0, tmp);
                k0 = vshrq_n_u16(k0, 3);

                k3 = vandq_u16(k3, mask);
                k2 = vandq_u16(k2, mask);
                k1 = vandq_u16(k1, mask);
                k0 = vandq_u16(k0, mask);
                vst1q_u16(&dst3[addr], k3);
                vst1q_u16(&dst2[addr], k2);
                vst1q_u16(&dst1[addr], k1);
                vst1q_u16(&dst0[addr], k0);
             }

}

/*
Because the input has only degree 677, omit the calculation of 680-720

w points to 9 (ordered) output size-144 vectors,
polynomial points to 1 input size-720 vector
*/
void tc5(uint16_t *restrict w, uint16_t *restrict polynomial) {
    uint16_t *w0_mem = &w[0*SB0],
             *w1_mem = &w[1*SB0],
             *w2_mem = &w[2*SB0],
             *w3_mem = &w[3*SB0],
             *w4_mem = &w[4*SB0],
             *w5_mem = &w[5*SB0],
             *w6_mem = &w[6*SB0],
             *w7_mem = &w[7*SB0],
             *w8_mem = &w[8*SB0],
             *c0 = &polynomial[0*SB0],
             *c1 = &polynomial[1*SB0],
             *c2 = &polynomial[2*SB0],
             *c3 = &polynomial[3*SB0],
             *c4 = &polynomial[4*SB0];
    uint16x8_t r0, r1, r2, r3, r4, p0, p1, p_1, tp;
    uint16x8_t zero;
    zero = vmovq_n_u16(0);
    for (uint16_t addr = 0; addr < 8*13; addr+= 8){
        r0 = vld1q_u16(&c0[addr]);
        r1 = vld1q_u16(&c1[addr]);
        r2 = vld1q_u16(&c2[addr]);
        r3 = vld1q_u16(&c3[addr]);
        r4 = vld1q_u16(&c4[addr]);

        p0 = vaddq_u16(r0, r2);  // p0  = r0 + r2
        p0 = vaddq_u16(p0, r4);  // p0  = p0 + r4 = r0 + r2 + r4
        tp = vaddq_u16(r1, r3);  // tp  = r1 + r3

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = r0 + r2 + r4 + r1 + r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w0_mem[addr], r0); // A(0)   = r0
        vst1q_u16(&w1_mem[addr], p1); // A(1)   = r0 + r2 + r4 + r1 + r3
        vst1q_u16(&w2_mem[addr], p_1);// A(-1)  = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w8_mem[addr], r4); // A(inf) = r4

        // deal w/ A(2), A(-2)
        p0 = vshlq_n_u16(r4, 2);  // p0 = (4) *(r4)
        p0 = vaddq_u16(p0, r2); // p0 = (4) *(r4) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r4) + (4)*r2
        p0 = vaddq_u16(p0, r0); // p0 = (16)*(r4) + (4)*r2 + r0

        tp = vshlq_n_u16(r3,  2); // tp = (4)*(r3)
        tp = vaddq_u16(tp, r1); // tp = (4)*(r3) + r1
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r3) + (2)*r1

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1
        vst1q_u16(&w3_mem[addr], p1); // A(2)    = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        vst1q_u16(&w4_mem[addr], p_1);// A(-2)   = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1

        // deal w/ A(3)
        p0 = vmulq_n_u16(r4, 9);  // p0 = (9) *(r4)
        p0 = vaddq_u16(p0, r2); // p0 = (9) *(r4) + r2
        p0 = vmulq_n_u16(p0, 9);  // p0 = (81)*(r4) + (9)*r2
        p0 = vaddq_u16(p0, r0); // p0 = (81)*(r4) + (9)*r2 + r0

        tp = vmulq_n_u16(r3, 9);   // tp = (9)*(r3)
        tp = vaddq_u16(tp, r1);    // tp = (9)*(r3) + r1
        tp = vmulq_n_u16(tp, 3);   // tp = (27)*(r3) + (3)*r1

        p1 = vaddq_u16( p0, tp); // p1  = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1
        vst1q_u16(&w5_mem[addr], p1);  // A(3)    = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1

        // deal w/ A(1/2), A(-1/2)
        p0 = vshlq_n_u16(r0, 2);  // p0 = (4) *(r0)
        p0 = vaddq_u16(p0, r2); // p0 = (4) *(r0) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r0) + (4)*r2
        p0 = vaddq_u16(p0, r4); // p0 = (16)*(r0) + (4)*r2 + r4

        tp = vshlq_n_u16(r1,  2); // tp = (4)*(r1)
        tp = vaddq_u16(tp, r3); // tp = (4)*(r1) + r3
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r1) + (2)*r3

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3

        vst1q_u16(&w6_mem[addr], p1);  // A(1/2)   = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        vst1q_u16(&w7_mem[addr], p_1); // A(-1/2)  = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3
    }
    for (uint16_t addr = 8*13; addr < SB0; addr+= 8){
        r0 = vld1q_u16(&c0[addr]);
        r1 = vld1q_u16(&c1[addr]);
        r2 = vld1q_u16(&c2[addr]);
        r3 = vld1q_u16(&c3[addr]);
        //r4 = vld1q_u16(&c4[addr]);  //r4 = 0

        p0 = vaddq_u16(r0, r2);  // p0  = r0 + r2
        tp = vaddq_u16(r1, r3);  // tp  = r1 + r3

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = r0 + r2 + r4 + r1 + r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = r0 + r2 + r4 - r1 - r3
        vst1q_u16(&w0_mem[addr], r0); // A(0)   = r0
        vst1q_u16(&w1_mem[addr], p1); // A(1)   = r0 + r2 + r4 + r1 + r3
        vst1q_u16(&w2_mem[addr], p_1);// A(-1)  = r0 + r2 + r4 - r1 - r3

        vst1q_u16(&w8_mem[addr], zero); // A(inf) = r4

        // deal w/ A(2), A(-2)
        p0 = vshlq_n_u16(r2, 2);  // p0 = (16)*(r4) + (4)*r2
        p0 = vaddq_u16(p0, r0); // p0 = (16)*(r4) + (4)*r2 + r0

        tp = vshlq_n_u16(r3,  2); // tp = (4)*(r3)
        tp = vaddq_u16(tp, r1); // tp = (4)*(r3) + r1
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r3) + (2)*r1

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1
        vst1q_u16(&w3_mem[addr], p1); // A(2)    = (16)*(r4) + (4)*r2 + r0 + (8)*(r3) + (2)*r1
        vst1q_u16(&w4_mem[addr], p_1);// A(-2)   = (16)*(r4) + (4)*r2 + r0 - (8)*(r3) - (2)*r1

        // deal w/ A(3)
        p0 = vmulq_n_u16(r2, 9);  // p0 = (81)*(r4) + (9)*r2
        p0 = vaddq_u16(p0, r0); // p0 = (81)*(r4) + (9)*r2 + r0

        tp = vmulq_n_u16(r3, 9);   // tp = (9)*(r3)
        tp = vaddq_u16(tp, r1);    // tp = (9)*(r3) + r1
        tp = vmulq_n_u16(tp, 3);   // tp = (27)*(r3) + (3)*r1

        p1 = vaddq_u16( p0, tp); // p1  = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1
        vst1q_u16(&w5_mem[addr], p1);  // A(3)    = (81)*(r4) + (9)*r2 + r0 + (27)*(r3) + (3)*r1

        // deal w/ A(1/2), A(-1/2)
        p0 = vshlq_n_u16(r0, 2);  // p0 = (4) *(r0)
        p0 = vaddq_u16(p0, r2); // p0 = (4) *(r0) + r2
        p0 = vshlq_n_u16(p0, 2);  // p0 = (16)*(r0) + (4)*r2

        tp = vshlq_n_u16(r1,  2); // tp = (4)*(r1)
        tp = vaddq_u16(tp, r3); // tp = (4)*(r1) + r3
        tp = vshlq_n_u16(tp, 1);  // tp = (8)*(r1) + (2)*r3

        p1 = vaddq_u16( p0, tp); // p1  = p0 + tp = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        p_1 = vsubq_u16(p0, tp); // p_1 = p0 - tp = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3

        vst1q_u16(&w6_mem[addr], p1);  // A(1/2)   = (16)*(r0) + (4)*r2 + r4 + (8)*(r1) + (2)*r3
        vst1q_u16(&w7_mem[addr], p_1); // A(-1/2)  = (16)*(r0) + (4)*r2 + r4 - (8)*(r1) - (2)*r3
    }
}


/*
w points to 25 (ordered) output size-16 vectors,
src points to 1 input size-144 vector
*/
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
    for (uint16_t addr = 0; addr < SB2; addr+=8)
    {
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

/*
w points to 25 (ordered) input size-16 vectors,
src points to 1 output size-144 vector
*/

void ttc33(uint16_t *restrict src, uint16_t *restrict w){
    uint16_t *k0 = &src[0*SB2],
             *k1 = &src[1*SB2],
             *k2 = &src[2*SB2],
             *k3 = &src[3*SB2],
             *k4 = &src[4*SB2],
             *k5 = &src[5*SB2],
             *k6 = &src[6*SB2],
             *k7 = &src[7*SB2],
             *k8 = &src[8*SB2],
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

             uint16x8_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
             uint16x8_t p10, p11, p12, p13, p14, p15, p16, p17, p18, p19;
             uint16x8_t p20, p21, p22, p23, p24;
             uint16x8_t tmp;
             uint16x8_t c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;
             uint16x8_t c10, c11, c12, c13, c14;

             for (uint16_t addr = 0; addr < SB2; addr+= 8){
//P0
                p0 = vld1q_u16(&w00[addr]);
                p1 = vld1q_u16(&w01[addr]);
                p1 = vmulq_n_u16(p1, inv3);
                p2 = vld1q_u16(&w02[addr]);
                p3 = vld1q_u16(&w03[addr]);
                p3 = vmulq_n_u16(p3, inv3);
                p4 = vld1q_u16(&w04[addr]);

                tmp = vshlq_n_u16(p3, 2);
                tmp = vaddq_u16(tmp, p1);
                tmp = vaddq_u16(tmp, p2);
                c0 = vshrq_n_u16(tmp, 1);
                c0 = vaddq_u16(c0, p4);

                tmp = vsubq_u16(p1, p2);
                tmp = vshrq_n_u16(tmp, 1);
                c1 = vsubq_u16(tmp, p3);

                tmp = vaddq_u16(p1, p2);
                tmp = vaddq_u16(tmp, p3);
                tmp = vaddq_u16(tmp, p0);
                c2 = vshrq_n_u16(tmp, 1);


//P1
                p5 = vld1q_u16(&w05[addr]);
                p6 = vld1q_u16(&w06[addr]);
                p6 = vmulq_n_u16(p6, inv3);
                p7 = vld1q_u16(&w07[addr]);
                p8 = vld1q_u16(&w08[addr]);
                p8 = vmulq_n_u16(p8, inv3);
                p9 = vld1q_u16(&w09[addr]);

                tmp = vshlq_n_u16(p8, 2);
                tmp = vaddq_u16(tmp, p6);
                tmp = vaddq_u16(tmp, p7);
                c3 = vshrq_n_u16(tmp, 1);
                c3 = vaddq_u16(c3, p9);

                tmp = vsubq_u16(p6, p7);
                tmp = vshrq_n_u16(tmp, 1);
                c4 = vsubq_u16(tmp, p8);

                tmp = vaddq_u16(p6, p7);
                tmp = vaddq_u16(tmp, p8);
                tmp = vaddq_u16(tmp, p5);
                c5 = vshrq_n_u16(tmp, 1);


//P2
                p10 = vld1q_u16(&w10[addr]);
                p11 = vld1q_u16(&w11[addr]);
                p11 = vmulq_n_u16(p11, inv3);
                p12 = vld1q_u16(&w12[addr]);
                p13 = vld1q_u16(&w13[addr]);
                p13 = vmulq_n_u16(p13, inv3);
                p14 = vld1q_u16(&w14[addr]);

                tmp = vshlq_n_u16(p13, 2);
                tmp = vaddq_u16(tmp, p11);
                tmp = vaddq_u16(tmp, p12);
                c6 = vshrq_n_u16(tmp, 1);
                c6 = vaddq_u16(c6, p14);

                tmp = vsubq_u16(p11, p12);
                tmp = vshrq_n_u16(tmp, 1);
                c7 = vsubq_u16(tmp, p13);

                tmp = vaddq_u16(p11, p12);
                tmp = vaddq_u16(tmp, p13);
                tmp = vaddq_u16(tmp, p10);
                c8 = vshrq_n_u16(tmp, 1);


//P3
                p15 = vld1q_u16(&w15[addr]);
                p16 = vld1q_u16(&w16[addr]);
                p16 = vmulq_n_u16(p16, inv3);
                p17 = vld1q_u16(&w17[addr]);
                p18 = vld1q_u16(&w18[addr]);
                p18 = vmulq_n_u16(p18, inv3);
                p19 = vld1q_u16(&w19[addr]);

                tmp = vshlq_n_u16(p18, 2);
                tmp = vaddq_u16(tmp, p16);
                tmp = vaddq_u16(tmp, p17);
                c9 = vshrq_n_u16(tmp, 1);
                c9 = vaddq_u16(c9, p19);

                tmp = vsubq_u16(p16, p17);
                tmp = vshrq_n_u16(tmp, 1);
                c10 = vsubq_u16(tmp, p18);

                tmp = vaddq_u16(p16, p17);
                tmp = vaddq_u16(tmp, p18);
                tmp = vaddq_u16(tmp, p15);
                c11 = vshrq_n_u16(tmp, 1);


//P4
                p20 = vld1q_u16(&w20[addr]);
                p21 = vld1q_u16(&w21[addr]);
                p21 = vmulq_n_u16(p21, inv3);
                p22 = vld1q_u16(&w22[addr]);
                p23 = vld1q_u16(&w23[addr]);
                p23 = vmulq_n_u16(p23, inv3);
                p24 = vld1q_u16(&w24[addr]);

                tmp = vshlq_n_u16(p23, 2);
                tmp = vaddq_u16(tmp, p21);
                tmp = vaddq_u16(tmp, p22);
                c12 = vshrq_n_u16(tmp, 1);
                c12 = vaddq_u16(c12, p24);

                tmp = vsubq_u16(p21, p22);
                tmp = vshrq_n_u16(tmp, 1);
                c13 = vsubq_u16(tmp, p23);

                tmp = vaddq_u16(p21, p22);
                tmp = vaddq_u16(tmp, p23);
                tmp = vaddq_u16(tmp, p20);
                c14 = vshrq_n_u16(tmp, 1);


//P1, P3 div by 3
                c3 = vmulq_n_u16(c3, inv3);
                c4 = vmulq_n_u16(c4, inv3);
                c5 = vmulq_n_u16(c5, inv3);

                c9 = vmulq_n_u16(c9, inv3);
                c10 = vmulq_n_u16(c10, inv3);
                c11 = vmulq_n_u16(c11, inv3);


//K0
                tmp = vshlq_n_u16(c9, 2);
                tmp = vaddq_u16(tmp, c3);
                tmp = vaddq_u16(tmp, c6);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vaddq_u16(tmp, c12);
                vst1q_u16(&k0[addr], tmp);

                tmp = vsubq_u16(c3, c6);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vsubq_u16(tmp, c9);
                vst1q_u16(&k3[addr], tmp);

                tmp = vaddq_u16(c3, c6);
                tmp = vaddq_u16(tmp, c9);
                tmp = vaddq_u16(tmp, c0);
                tmp = vshrq_n_u16(tmp, 1);
                vst1q_u16(&k6[addr], tmp);


//K1
                tmp = vshlq_n_u16(c10, 2);
                tmp = vaddq_u16(tmp, c4);
                tmp = vaddq_u16(tmp, c7);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vaddq_u16(tmp, c13);
                vst1q_u16(&k1[addr], tmp);

                tmp = vsubq_u16(c4, c7);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vsubq_u16(tmp, c10);
                vst1q_u16(&k4[addr], tmp);

                tmp = vaddq_u16(c4, c7);
                tmp = vaddq_u16(tmp, c10);
                tmp = vaddq_u16(tmp, c1);
                tmp = vshrq_n_u16(tmp, 1);
                vst1q_u16(&k7[addr], tmp);


//K2
                tmp = vshlq_n_u16(c11, 2);
                tmp = vaddq_u16(tmp, c5);
                tmp = vaddq_u16(tmp, c8);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vaddq_u16(tmp, c14);
                vst1q_u16(&k2[addr], tmp);

                tmp = vsubq_u16(c5, c8);
                tmp = vshrq_n_u16(tmp, 1);
                tmp = vsubq_u16(tmp, c11);
                vst1q_u16(&k5[addr], tmp);

                tmp = vaddq_u16(c5, c8);
                tmp = vaddq_u16(tmp, c11);
                tmp = vaddq_u16(tmp, c2);
                tmp = vshrq_n_u16(tmp, 1);
                vst1q_u16(&k8[addr], tmp);

             }
}

/*
toeplitz matrix:
    tmvp3_split -> tmvp32_split -> mixed schoolbook
input vector:
    tmvp33_split -> mixed schoolbook

mixed schoolbook:
    vertor: tmvp2_split -> schoolbook

schoolbook: 8*8 toeplitz matrix to vector -> tmvp2_combine

output vector: tmvp33_combine
*/
void tmvp33(uint16_t *restrict polyC, uint16_t *restrict toepA, uint16_t *restrict polyB) {
    uint16_t toepa3[SB1 * 5 * 2]; // SB1 = 48

    uint16_t tmp[5 * 5 * SB2 * 4];

    uint16_t *toepa332 = &tmp[5 * 5 * SB2 * 0];
    uint16_t *kbcw = &tmp[5 * 5 * SB2 * 3];

    ittc3(toepa3, toepA); /* 1.8k cycles, 1.8k/9 = 0.2k each */
    ittc32(toepa332, toepa3); /* 5k cycles, 5k/9 = 0.5k each */

    tc33(kbcw, polyB); /* 1.3k cycles, 1.3k/9 = 0.1k each */

    tmvp2_8x8(kbcw, toepa332); /* 12.6k cycles, 12.6k/9 = 1.4k each */

    ttc33(polyC, kbcw); /* 1.8k cycles, 1.8k/9 = 0.2k each */
}

void tmvp33_last(uint16_t *restrict polyC, uint16_t *restrict toepA, uint16_t *restrict polyB) {
    uint16_t toepa3[SB1 * 5 * 2]; // SB1 = 48

    uint16_t tmp[5 * 5 * SB2 * 4];

    uint16_t *toepa332 = &tmp[5 * 5 * SB2 * 0];
    uint16_t *kbcw = &tmp[5 * 5 * SB2 * 3];



    ittc3(toepa3, toepA); /* 1.8k cycles, 1.8k/9 = 0.2k each */
    ittc32(toepa332, toepa3); /* 5k cycles, 5k/9 = 0.5k each */

    tc33(kbcw, polyB); /* 1.3k cycles, 1.3k/9 = 0.1k each */

    tmvp2_8x8(kbcw, toepa332); /* 12.6k cycles, 12.6k/9 = 1.4k each */

/* Let gcc know the last schoolbook is all zero */
    for(int i = 0; i < 16; i++){
        kbcw[24*16+i] = 0;
    }

    ttc33(polyC, kbcw); /* 1.8k cycles, 1.8k/9 = 0.2k each */
}

/*
tmvp5332 method.
1 tmvp5, and 9 tmvp332.
The last 16 coefficients of the last tmvp332 is unused.
*/
void tmvp(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB){
    uint16_t tmp[SB0 * 9 * 4]; // SB0 = 144

    uint16_t *toepa = &tmp[0 * SB0]; /* nine 144*144 toeplitz matrix, needs nine length-288 vectors to store */
    uint16_t *kbw   = &tmp[18* SB0]; /* nine 144*144 vectors*/
    uint16_t *kcw   = &tmp[27* SB0]; /* nine 144*144 vectors*/

    ittc5(toepa, polyA); /* 2.5k cycles */

    tc5(kbw, polyB); /* 0.6k cycles */


    for(int i = 0; i < 8; i++){
        tmvp33(&kcw[i * SB0], &toepa[i * SB0 * 2], &kbw[i * SB0]);
    }

    tmvp33_last(&kcw[8 * SB0], &toepa[8 * SB0 * 2], &kbw[8 * SB0]);

    ttc5(polyC, kcw); /* 0.8k cycles */

}





