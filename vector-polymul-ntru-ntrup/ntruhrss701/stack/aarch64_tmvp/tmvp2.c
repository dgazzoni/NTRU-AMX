
#include <stdio.h>
#include <arm_neon.h>
#include <stdlib.h>
#include "params.h"
#include "poly.h"

#include "tmvp.h"

// Modified in NTRU-AMX to remove unused header
#if NN==701
#include "cpucycles.h"
#endif

static const unsigned int ZREV[4]  = {0xffffffff, 0x0100ffff, 0x05040302, 0x09080706};

/*
Because the input has only degree 701, omit the calculation of 704-720

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


            for (uint16_t addr = 0; addr < 8*2; addr+= 8) {
                z0 = vld1q_u16(&polynomial[701-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[701-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[701-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[701-SB0+addr]);
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
             for (uint16_t addr = 8*2; addr < 8*3; addr+= 8) {
                z0 = vld1q_u16(&polynomial[701-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[701-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[701-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[701-SB0+addr]);
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
             for (uint16_t addr = 8*3; addr < 8*16; addr+= 8) {
                z0 = vld1q_u16(&polynomial[701-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[701-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[701-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[701-SB0+addr]);
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




                z0 = vld1q_u16(&polynomial[701-SB0*5+addr]);
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
             for (uint16_t addr = 8*16; addr < SB0; addr+= 8) {
                z0 = vld1q_u16(&polynomial[701-SB0*4+addr]);
                z1 = vld1q_u16(&polynomial[701-SB0*3+addr]);
                z2 = vld1q_u16(&polynomial[701-SB0*2+addr]);
                z3 = vld1q_u16(&polynomial[701-SB0+addr]);
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




                z0 = vld1q_u16(&polynomial[701-SB0*5+addr]);
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
Because the output has only degree 701, omit the calculation of 704-720

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

             for (uint16_t addr = 0; addr < 8*16; addr+= 8){
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
             for (uint16_t addr = 8*16; addr < SB0; addr+= 8){
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
Because the input has only degree 701, omit the calculation of 704-720

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
    for (uint16_t addr = 0; addr < 8*16; addr+= 8){
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
    for (uint16_t addr = 8*16; addr < SB0; addr+= 8){
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
toeplitz matrix:
    tmvp3_split -> tmvp32_split -> mixed schoolbook
input vector:
    tmvp33_split -> mixed schoolbook

mixed schoolbook:
    vertor: tmvp2_split -> schoolbook

schoolbook: 8*8 toeplitz matrix to vector -> tmvp2_combine

output vector: tmvp33_combine
*/
void tmvp_16x16_x2_ka(uint16_t  *VecB, uint16_t  *restrict ToepA, uint16_t  *vecb, uint16_t  *restrict toepa) {

  uint16x8_t A0, A1, A2, A3, B0, B1;
  uint16x8_t TA0, TA1, TA2, TB, TC0, TC1, TC2, T0, T1, T2;
  uint16x8_t a0, a1, a2, a3, b0, b1;
  uint16x8_t ta0, ta1, ta2, tb, tc0, tc1, tc2, t0, t1, t2;
  // uint16_t  *vecb = &VecB[16];
  // uint16_t  *restrict toepa = &ToepA[32];

  A0 = vld1q_u16(ToepA)   ; A1 = vld1q_u16(ToepA+8);
  A2 = vld1q_u16(ToepA+16); A3 = vld1q_u16(ToepA+24);
  B0 = vld1q_u16(VecB)    ; B1 = vld1q_u16(VecB+8);
  TA2 = vsubq_u16(A3,A2)  ; TA1 = vsubq_u16(A2,A1);
  TA0 = vsubq_u16(A1,A0)  ; TB = vaddq_u16(B0,B1);

  a0 = vld1q_u16(toepa)   ; a1 = vld1q_u16(toepa+8);
  a2 = vld1q_u16(toepa+16); a3 = vld1q_u16(toepa+24);
  b0 = vld1q_u16(vecb)    ; b1 = vld1q_u16(vecb+8);
  ta2 = vsubq_u16(a3,a2)  ; ta1 = vsubq_u16(a2,a1);
  ta0 = vsubq_u16(a1,a0)  ; tb = vaddq_u16(b0,b1);
  
  // compute TC0 = (TA1, TA2) x B0
  // compute TC1 = (TA0, TA1) x B1
  // compute TC2 = (A1, A2) x TB
 
  TC0 = vmulq_laneq_u16(TA2, B0, 0);
  TC1 = vmulq_laneq_u16(TA1, B1, 0);
  TC2 = vmulq_laneq_u16(A2, TB, 0);
  tc0 = vmulq_laneq_u16(ta2, b0, 0);
  tc1 = vmulq_laneq_u16(ta1, b1, 0);
  tc2 = vmulq_laneq_u16(a2, tb, 0);
  
  T0 = vextq_u16(TA1, TA2, 7);
  T1 = vextq_u16(TA0, TA1, 7);
  T2 = vextq_u16(A1, A2, 7);
  t0 = vextq_u16(ta1, ta2, 7);
  t1 = vextq_u16(ta0, ta1, 7);
  t2 = vextq_u16(a1, a2, 7);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 1);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 1);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 1);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 1);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 1);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 1);

  T0 = vextq_u16(TA1, TA2, 6);
  T1 = vextq_u16(TA0, TA1, 6);
  T2 = vextq_u16(A1, A2, 6);
  t0 = vextq_u16(ta1, ta2, 6);
  t1 = vextq_u16(ta0, ta1, 6);
  t2 = vextq_u16(a1, a2, 6);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 2);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 2);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 2);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 2);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 2);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 2);

  T0 = vextq_u16(TA1, TA2, 5);
  T1 = vextq_u16(TA0, TA1, 5);
  T2 = vextq_u16(A1, A2, 5);
  t0 = vextq_u16(ta1, ta2, 5);
  t1 = vextq_u16(ta0, ta1, 5);
  t2 = vextq_u16(a1, a2, 5);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 3);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 3);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 3);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 3);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 3);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 3);

  T0 = vextq_u16(TA1, TA2, 4);
  T1 = vextq_u16(TA0, TA1, 4);
  T2 = vextq_u16(A1, A2, 4);
  t0 = vextq_u16(ta1, ta2, 4);
  t1 = vextq_u16(ta0, ta1, 4);
  t2 = vextq_u16(a1, a2, 4);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 4);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 4);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 4);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 4);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 4);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 4);

  T0 = vextq_u16(TA1, TA2, 3);
  T1 = vextq_u16(TA0, TA1, 3);
  T2 = vextq_u16(A1, A2, 3);
  t0 = vextq_u16(ta1, ta2, 3);
  t1 = vextq_u16(ta0, ta1, 3);
  t2 = vextq_u16(a1, a2, 3);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 5);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 5);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 5);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 5);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 5);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 5);

  T0 = vextq_u16(TA1, TA2, 2);
  T1 = vextq_u16(TA0, TA1, 2);
  T2 = vextq_u16(A1, A2, 2);
  t0 = vextq_u16(ta1, ta2, 2);
  t1 = vextq_u16(ta0, ta1, 2);
  t2 = vextq_u16(a1, a2, 2);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 6);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 6);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 6);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 6);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 6);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 6);

  T0 = vextq_u16(TA1, TA2, 1);
  T1 = vextq_u16(TA0, TA1, 1);
  T2 = vextq_u16(A1, A2, 1);
  t0 = vextq_u16(ta1, ta2, 1);
  t1 = vextq_u16(ta0, ta1, 1);
  t2 = vextq_u16(a1, a2, 1);

  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 7);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 7);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 7);
  tc0 = vmlaq_laneq_u16(tc0, t0, b0, 7);
  tc1 = vmlaq_laneq_u16(tc1, t1, b1, 7);
  tc2 = vmlaq_laneq_u16(tc2, t2, tb, 7);

  T0 = vsubq_u16(TC2, TC1);
  T1 = vaddq_u16(TC2, TC0);
  t0 = vsubq_u16(tc2, tc1);
  t1 = vaddq_u16(tc2, tc0);

  vst1q_u16(VecB,T0);
  vst1q_u16(VecB+8,T1);
  vst1q_u16(vecb,t0);
  vst1q_u16(vecb+8,t1);
}


#define _6_to_12(A00,A01,A10,A11,A20,A21,A30,A31,			\
  A40,A41,A50,A51,T00,T01,T10,T11,P)                                    \
  vst1q_u16((P)+8*12,(A30)); vst1q_u16((P)+8*13,(A31));                 \
  vst1q_u16((P)+8*14,(A40)); vst1q_u16((P)+8*15,(A41));                 \
  vst1q_u16((P)+8*16,(A20)); vst1q_u16((P)+8*17,(A21));                 \
  vst1q_u16((P)+8*18,(A30)); vst1q_u16((P)+8*19,(A31));                 \
  vst1q_u16((P)+8*20,(A10)); vst1q_u16((P)+8*21,(A11));                 \
  vst1q_u16((P)+8*22,(A20)); vst1q_u16((P)+8*23,(A21));                 \
  T00 = vsubq_u16((A40),(A20)); T01 = vsubq_u16((A41),(A21));           \
  T10 = vsubq_u16((A50),(A30)); T11 = vsubq_u16((A51),(A31));           \
  T00 = vsubq_u16((T00),(A30)); T01 = vsubq_u16((T01),(A31));           \
  T10 = vsubq_u16((T10),(A40)); T11 = vsubq_u16((T11),(A41));           \
  vst1q_u16((P)+8*0,(T00)); vst1q_u16((P)+8*1,(T01));                   \
  vst1q_u16((P)+8*2,(T10)); vst1q_u16((P)+8*3,(T11));                   \
  T00 = vsubq_u16((A20),(A10)); T01 = vsubq_u16((A21),(A11));           \
  T10 = vsubq_u16((A30),(A20)); T11 = vsubq_u16((A31),(A21));           \
  T00 = vsubq_u16((T00),(A30)); T01 = vsubq_u16((T01),(A31));           \
  T10 = vsubq_u16((T10),(A40)); T11 = vsubq_u16((T11),(A41));           \
  vst1q_u16((P)+8*4,(T00)); vst1q_u16((P)+8*5,(T01));                   \
  vst1q_u16((P)+8*6,(T10)); vst1q_u16((P)+8*7,(T11));                   \
  T00 = vsubq_u16((A00),(A10)); T01 = vsubq_u16((A01),(A11));           \
  T10 = vsubq_u16((A10),(A20)); T11 = vsubq_u16((A11),(A21));           \
  T00 = vsubq_u16((T00),(A20)); T01 = vsubq_u16((T01),(A21));           \
  T10 = vsubq_u16((T10),(A30)); T11 = vsubq_u16((T11),(A31));           \
  vst1q_u16((P)+8*8,(T00)); vst1q_u16((P)+8*9,(T01));                   \
  vst1q_u16((P)+8*10,(T10)); vst1q_u16((P)+8*11,(T11));                 \
  // expand six 16xu16 to twelve 16xu16

#define _3_to_6(B00,B01,B10,B11,B20,B21,T0,T1,P)                     \
  vst1q_u16((P)+8*0,(B00)); vst1q_u16((P)+8*1,(B01));                   \
  vst1q_u16((P)+8*2,(B10)); vst1q_u16((P)+8*3,(B11));                   \
  vst1q_u16((P)+8*4,(B20)); vst1q_u16((P)+8*5,(B21));                   \
  T0 = vaddq_u16((B00),(B10)); T1 = vaddq_u16((B01),(B11));             \
  vst1q_u16((P)+8*6,(T0)); vst1q_u16((P)+8*7,(T1));                     \
  T0 = vaddq_u16((B00),(B20)); T1 = vaddq_u16((B01),(B21));             \
  vst1q_u16((P)+8*8,(T0)); vst1q_u16((P)+8*9,(T1));                     \
  T0 = vaddq_u16((B10),(B20)); T1 = vaddq_u16((B11),(B21));             \
  vst1q_u16((P)+8*10,(T0)); vst1q_u16((P)+8*11,(T1));                     \
  // expand 3 16xu16 to twelve 16xu16
  



void tmvp_144_ka33_ka2(uint16_t *VecB, uint16_t *restrict ToepA){
  uint16_t TmpB[576], TmpA[1152];
  uint16x8_t A[36], TA[36], B[18], TB[18], T[6];
  int i;

  A[0]=vld1q_u16(ToepA+8*0);
  A[1]=vld1q_u16(ToepA+8*1);
  A[2]=vld1q_u16(ToepA+8*2);
  A[3]=vld1q_u16(ToepA+8*3);
  A[4]=vld1q_u16(ToepA+8*4);
  A[5]=vld1q_u16(ToepA+8*5);
  A[6]=vld1q_u16(ToepA+8*6);
  A[7]=vld1q_u16(ToepA+8*7);
  A[8]=vld1q_u16(ToepA+8*8);
  A[9]=vld1q_u16(ToepA+8*9);
  A[10]=vld1q_u16(ToepA+8*10);
  A[11]=vld1q_u16(ToepA+8*11);
  A[12]=vld1q_u16(ToepA+8*12);
  A[13]=vld1q_u16(ToepA+8*13);
  A[14]=vld1q_u16(ToepA+8*14);
  A[15]=vld1q_u16(ToepA+8*15);
  A[16]=vld1q_u16(ToepA+8*16);
  A[17]=vld1q_u16(ToepA+8*17);
  A[18]=vld1q_u16(ToepA+8*18);
  A[19]=vld1q_u16(ToepA+8*19);
  A[20]=vld1q_u16(ToepA+8*20);
  A[21]=vld1q_u16(ToepA+8*21);
  A[22]=vld1q_u16(ToepA+8*22);
  A[23]=vld1q_u16(ToepA+8*23);
  _6_to_12(A[6],A[7],A[8],A[9],A[10],A[11],A[12],A[13],A[14],A[15],A[16],A[17],T[0],T[1],T[2],T[3],TmpA+192*5);
  _6_to_12(A[12],A[13],A[14],A[15],A[16],A[17],A[18],A[19],A[20],A[21],A[22],A[23],T[0],T[1],T[2],T[3],TmpA+192*4);
  TA[0]=vsubq_u16(A[0],A[6]);
  TA[1]=vsubq_u16(A[1],A[7]);
  TA[2]=vsubq_u16(A[2],A[8]);
  TA[3]=vsubq_u16(A[3],A[9]);
  TA[4]=vsubq_u16(A[4],A[10]);
  TA[5]=vsubq_u16(A[5],A[11]);
  TA[6]=vsubq_u16(A[6],A[12]);
  TA[7]=vsubq_u16(A[7],A[13]);
  TA[8]=vsubq_u16(A[8],A[14]);
  TA[9]=vsubq_u16(A[9],A[15]);
  TA[10]=vsubq_u16(A[10],A[16]);
  TA[11]=vsubq_u16(A[11],A[17]);
  TA[0]=vsubq_u16(TA[0],A[12]);
  TA[1]=vsubq_u16(TA[1],A[13]);
  TA[2]=vsubq_u16(TA[2],A[14]);
  TA[3]=vsubq_u16(TA[3],A[15]);
  TA[4]=vsubq_u16(TA[4],A[16]);
  TA[5]=vsubq_u16(TA[5],A[17]);
  TA[6]=vsubq_u16(TA[6],A[18]);
  TA[7]=vsubq_u16(TA[7],A[19]);
  TA[8]=vsubq_u16(TA[8],A[20]);
  TA[9]=vsubq_u16(TA[9],A[21]);
  TA[10]=vsubq_u16(TA[10],A[22]);
  TA[11]=vsubq_u16(TA[11],A[23]);
  _6_to_12(TA[0],TA[1],TA[2],TA[3],TA[4],TA[5],TA[6],TA[7],TA[8],TA[9],TA[10],TA[11],T[0],T[1],T[2],T[3],TmpA+192*2);
  A[24]=vld1q_u16(ToepA+8*24);
  A[25]=vld1q_u16(ToepA+8*25);
  A[26]=vld1q_u16(ToepA+8*26);
  A[27]=vld1q_u16(ToepA+8*27);
  A[28]=vld1q_u16(ToepA+8*28);
  A[29]=vld1q_u16(ToepA+8*29);
  _6_to_12(A[18],A[19],A[20],A[21],A[22],A[23],A[24],A[25],A[26],A[27],A[28],A[29],T[0],T[1],T[2],T[3],TmpA+192*3);
  TA[0]=vsubq_u16(A[12],A[6]);
  TA[1]=vsubq_u16(A[13],A[7]);
  TA[2]=vsubq_u16(A[14],A[8]);
  TA[3]=vsubq_u16(A[15],A[9]);
  TA[4]=vsubq_u16(A[16],A[10]);
  TA[5]=vsubq_u16(A[17],A[11]);
  TA[6]=vsubq_u16(A[18],A[12]);
  TA[7]=vsubq_u16(A[19],A[13]);
  TA[8]=vsubq_u16(A[20],A[14]);
  TA[9]=vsubq_u16(A[21],A[15]);
  TA[10]=vsubq_u16(A[22],A[16]);
  TA[11]=vsubq_u16(A[23],A[17]);
  TA[0]=vsubq_u16(TA[0],A[18]);
  TA[1]=vsubq_u16(TA[1],A[19]);
  TA[2]=vsubq_u16(TA[2],A[20]);
  TA[3]=vsubq_u16(TA[3],A[21]);
  TA[4]=vsubq_u16(TA[4],A[22]);
  TA[5]=vsubq_u16(TA[5],A[23]);
  TA[6]=vsubq_u16(TA[6],A[24]);
  TA[7]=vsubq_u16(TA[7],A[25]);
  TA[8]=vsubq_u16(TA[8],A[26]);
  TA[9]=vsubq_u16(TA[9],A[27]);
  TA[10]=vsubq_u16(TA[10],A[28]);
  TA[11]=vsubq_u16(TA[11],A[29]);
  _6_to_12(TA[0],TA[1],TA[2],TA[3],TA[4],TA[5],TA[6],TA[7],TA[8],TA[9],TA[10],TA[11],T[0],T[1],T[2],T[3],TmpA+192*1);
  A[30]=vld1q_u16(ToepA+8*30);
  A[31]=vld1q_u16(ToepA+8*31);
  A[32]=vld1q_u16(ToepA+8*32);
  A[33]=vld1q_u16(ToepA+8*33);
  A[34]=vld1q_u16(ToepA+8*34);
  A[35]=vld1q_u16(ToepA+8*35);
  TA[0]=vsubq_u16(A[24],A[18]);
  TA[1]=vsubq_u16(A[25],A[19]);
  TA[2]=vsubq_u16(A[26],A[20]);
  TA[3]=vsubq_u16(A[27],A[21]);
  TA[4]=vsubq_u16(A[28],A[22]);
  TA[5]=vsubq_u16(A[29],A[23]);
  TA[6]=vsubq_u16(A[30],A[24]);
  TA[7]=vsubq_u16(A[31],A[25]);
  TA[8]=vsubq_u16(A[32],A[26]);
  TA[9]=vsubq_u16(A[33],A[27]);
  TA[10]=vsubq_u16(A[34],A[28]);
  TA[11]=vsubq_u16(A[35],A[29]);
  TA[0]=vsubq_u16(TA[0],A[12]);
  TA[1]=vsubq_u16(TA[1],A[13]);
  TA[2]=vsubq_u16(TA[2],A[14]);
  TA[3]=vsubq_u16(TA[3],A[15]);
  TA[4]=vsubq_u16(TA[4],A[16]);
  TA[5]=vsubq_u16(TA[5],A[17]);
  TA[6]=vsubq_u16(TA[6],A[18]);
  TA[7]=vsubq_u16(TA[7],A[19]);
  TA[8]=vsubq_u16(TA[8],A[20]);
  TA[9]=vsubq_u16(TA[9],A[21]);
  TA[10]=vsubq_u16(TA[10],A[22]);
  TA[11]=vsubq_u16(TA[11],A[23]);
  _6_to_12(TA[0],TA[1],TA[2],TA[3],TA[4],TA[5],TA[6],TA[7],TA[8],TA[9],TA[10],TA[11],T[0],T[1],T[2],T[3],TmpA+192*0);
  // TmpA set
  B[0]=vld1q_u16(VecB+8*0);
  B[1]=vld1q_u16(VecB+8*1);
  B[2]=vld1q_u16(VecB+8*2);
  B[3]=vld1q_u16(VecB+8*3);
  B[4]=vld1q_u16(VecB+8*4);
  B[5]=vld1q_u16(VecB+8*5);
  B[6]=vld1q_u16(VecB+8*6);
  B[7]=vld1q_u16(VecB+8*7);
  B[8]=vld1q_u16(VecB+8*8);
  B[9]=vld1q_u16(VecB+8*9);
  B[10]=vld1q_u16(VecB+8*10);
  B[11]=vld1q_u16(VecB+8*11);
  B[12]=vld1q_u16(VecB+8*12);
  B[13]=vld1q_u16(VecB+8*13);
  B[14]=vld1q_u16(VecB+8*14);
  B[15]=vld1q_u16(VecB+8*15);
  B[16]=vld1q_u16(VecB+8*16);
  B[17]=vld1q_u16(VecB+8*17);
  _3_to_6(B[0],B[1],B[2],B[3],B[4],B[5],T[0],T[1],TmpB+96*0);
  _3_to_6(B[6],B[7],B[8],B[9],B[10],B[11],T[0],T[1],TmpB+96*1);
  _3_to_6(B[12],B[13],B[14],B[15],B[16],B[17],T[0],T[1],TmpB+96*2);
  TB[0]=vaddq_u16(B[0],B[6]);
  TB[1]=vaddq_u16(B[1],B[7]);
  TB[2]=vaddq_u16(B[2],B[8]);
  TB[3]=vaddq_u16(B[3],B[9]);
  TB[4]=vaddq_u16(B[4],B[10]);
  TB[5]=vaddq_u16(B[5],B[11]);
  TB[6]=vaddq_u16(B[0],B[12]);
  TB[7]=vaddq_u16(B[1],B[13]);
  TB[8]=vaddq_u16(B[2],B[14]);
  TB[9]=vaddq_u16(B[3],B[15]);
  TB[10]=vaddq_u16(B[4],B[16]);
  TB[11]=vaddq_u16(B[5],B[17]);
  TB[12]=vaddq_u16(B[6],B[12]);
  TB[13]=vaddq_u16(B[7],B[13]);
  TB[14]=vaddq_u16(B[8],B[14]);
  TB[15]=vaddq_u16(B[9],B[15]);
  TB[16]=vaddq_u16(B[10],B[16]);
  TB[17]=vaddq_u16(B[11],B[17]);
  _3_to_6(TB[0],TB[1],TB[2],TB[3],TB[4],TB[5],T[0],T[1],TmpB+96*3);
  _3_to_6(TB[6],TB[7],TB[8],TB[9],TB[10],TB[11],T[0],T[1],TmpB+96*4);
  _3_to_6(TB[12],TB[13],TB[14],TB[15],TB[16],TB[17],T[0],T[1],TmpB+96*5);
  // TmpB set
  for(i=0; i<18; i++) tmvp_16x16_x2_ka(TmpB+32*i,TmpA+64*i,TmpB+32*i+16,TmpA+64*i+32);
  B[0]=vld1q_u16(TmpB+8*0);
  B[1]=vld1q_u16(TmpB+8*1);
  B[2]=vld1q_u16(TmpB+8*2);
  B[3]=vld1q_u16(TmpB+8*3);
  B[4]=vld1q_u16(TmpB+8*4);
  B[5]=vld1q_u16(TmpB+8*5);
  T[0]=vld1q_u16(TmpB+8*6);
  T[1]=vld1q_u16(TmpB+8*7);
  T[2]=vld1q_u16(TmpB+8*8);
  T[3]=vld1q_u16(TmpB+8*9);
  T[4]=vld1q_u16(TmpB+8*10);
  T[5]=vld1q_u16(TmpB+8*11);
  B[0]=vaddq_u16(B[0],T[0]);
  B[0]=vaddq_u16(B[0],T[2]);
  B[2]=vaddq_u16(B[2],T[0]);
  B[2]=vaddq_u16(B[2],T[4]);
  B[4]=vaddq_u16(B[4],T[2]);
  B[4]=vaddq_u16(B[4],T[4]);
  B[1]=vaddq_u16(B[1],T[1]);
  B[1]=vaddq_u16(B[1],T[3]);
  B[3]=vaddq_u16(B[3],T[1]);
  B[3]=vaddq_u16(B[3],T[5]);
  B[5]=vaddq_u16(B[5],T[3]);
  B[5]=vaddq_u16(B[5],T[5]);
  B[6]=vld1q_u16(TmpB+8*12);
  B[7]=vld1q_u16(TmpB+8*13);
  B[8]=vld1q_u16(TmpB+8*14);
  B[9]=vld1q_u16(TmpB+8*15);
  B[10]=vld1q_u16(TmpB+8*16);
  B[11]=vld1q_u16(TmpB+8*17);
  T[0]=vld1q_u16(TmpB+8*18);
  T[1]=vld1q_u16(TmpB+8*19);
  T[2]=vld1q_u16(TmpB+8*20);
  T[3]=vld1q_u16(TmpB+8*21);
  T[4]=vld1q_u16(TmpB+8*22);
  T[5]=vld1q_u16(TmpB+8*23);
  B[6]=vaddq_u16(B[6],T[0]);
  B[6]=vaddq_u16(B[6],T[2]);
  B[8]=vaddq_u16(B[8],T[0]);
  B[8]=vaddq_u16(B[8],T[4]);
  B[10]=vaddq_u16(B[10],T[2]);
  B[10]=vaddq_u16(B[10],T[4]);
  B[7]=vaddq_u16(B[7],T[1]);
  B[7]=vaddq_u16(B[7],T[3]);
  B[9]=vaddq_u16(B[9],T[1]);
  B[9]=vaddq_u16(B[9],T[5]);
  B[11]=vaddq_u16(B[11],T[3]);
  B[11]=vaddq_u16(B[11],T[5]);
  B[12]=vld1q_u16(TmpB+8*24);
  B[13]=vld1q_u16(TmpB+8*25);
  B[14]=vld1q_u16(TmpB+8*26);
  B[15]=vld1q_u16(TmpB+8*27);
  B[16]=vld1q_u16(TmpB+8*28);
  B[17]=vld1q_u16(TmpB+8*29);
  T[0]=vld1q_u16(TmpB+8*30);
  T[1]=vld1q_u16(TmpB+8*31);
  T[2]=vld1q_u16(TmpB+8*32);
  T[3]=vld1q_u16(TmpB+8*33);
  T[4]=vld1q_u16(TmpB+8*34);
  T[5]=vld1q_u16(TmpB+8*35);
  B[12]=vaddq_u16(B[12],T[0]);
  B[12]=vaddq_u16(B[12],T[2]);
  B[14]=vaddq_u16(B[14],T[0]);
  B[14]=vaddq_u16(B[14],T[4]);
  B[16]=vaddq_u16(B[16],T[2]);
  B[16]=vaddq_u16(B[16],T[4]);
  B[13]=vaddq_u16(B[13],T[1]);
  B[13]=vaddq_u16(B[13],T[3]);
  B[15]=vaddq_u16(B[15],T[1]);
  B[15]=vaddq_u16(B[15],T[5]);
  B[17]=vaddq_u16(B[17],T[3]);
  B[17]=vaddq_u16(B[17],T[5]);
  TB[0]=vld1q_u16(TmpB+8*36);
  TB[1]=vld1q_u16(TmpB+8*37);
  TB[2]=vld1q_u16(TmpB+8*38);
  TB[3]=vld1q_u16(TmpB+8*39);
  TB[4]=vld1q_u16(TmpB+8*40);
  TB[5]=vld1q_u16(TmpB+8*41);
  T[0]=vld1q_u16(TmpB+8*42);
  T[1]=vld1q_u16(TmpB+8*43);
  T[2]=vld1q_u16(TmpB+8*44);
  T[3]=vld1q_u16(TmpB+8*45);
  T[4]=vld1q_u16(TmpB+8*46);
  T[5]=vld1q_u16(TmpB+8*47);
  TB[0]=vaddq_u16(TB[0],T[0]);
  TB[0]=vaddq_u16(TB[0],T[2]);
  TB[2]=vaddq_u16(TB[2],T[0]);
  TB[2]=vaddq_u16(TB[2],T[4]);
  TB[4]=vaddq_u16(TB[4],T[2]);
  TB[4]=vaddq_u16(TB[4],T[4]);
  TB[1]=vaddq_u16(TB[1],T[1]);
  TB[1]=vaddq_u16(TB[1],T[3]);
  TB[3]=vaddq_u16(TB[3],T[1]);
  TB[3]=vaddq_u16(TB[3],T[5]);
  TB[5]=vaddq_u16(TB[5],T[3]);
  TB[5]=vaddq_u16(TB[5],T[5]);
  B[0]=vaddq_u16(B[0],TB[0]);
  B[6]=vaddq_u16(B[6],TB[0]);
  B[1]=vaddq_u16(B[1],TB[1]);
  B[7]=vaddq_u16(B[7],TB[1]);
  B[2]=vaddq_u16(B[2],TB[2]);
  B[8]=vaddq_u16(B[8],TB[2]);
  B[3]=vaddq_u16(B[3],TB[3]);
  B[9]=vaddq_u16(B[9],TB[3]);
  B[4]=vaddq_u16(B[4],TB[4]);
  B[10]=vaddq_u16(B[10],TB[4]);
  B[5]=vaddq_u16(B[5],TB[5]);
  B[11]=vaddq_u16(B[11],TB[5]);
  TB[6]=vld1q_u16(TmpB+8*48);
  TB[7]=vld1q_u16(TmpB+8*49);
  TB[8]=vld1q_u16(TmpB+8*50);
  TB[9]=vld1q_u16(TmpB+8*51);
  TB[10]=vld1q_u16(TmpB+8*52);
  TB[11]=vld1q_u16(TmpB+8*53);
  T[0]=vld1q_u16(TmpB+8*54);
  T[1]=vld1q_u16(TmpB+8*55);
  T[2]=vld1q_u16(TmpB+8*56);
  T[3]=vld1q_u16(TmpB+8*57);
  T[4]=vld1q_u16(TmpB+8*58);
  T[5]=vld1q_u16(TmpB+8*59);
  TB[6]=vaddq_u16(TB[6],T[0]);
  TB[6]=vaddq_u16(TB[6],T[2]);
  TB[8]=vaddq_u16(TB[8],T[0]);
  TB[8]=vaddq_u16(TB[8],T[4]);
  TB[10]=vaddq_u16(TB[10],T[2]);
  TB[10]=vaddq_u16(TB[10],T[4]);
  TB[7]=vaddq_u16(TB[7],T[1]);
  TB[7]=vaddq_u16(TB[7],T[3]);
  TB[9]=vaddq_u16(TB[9],T[1]);
  TB[9]=vaddq_u16(TB[9],T[5]);
  TB[11]=vaddq_u16(TB[11],T[3]);
  TB[11]=vaddq_u16(TB[11],T[5]);
  B[0]=vaddq_u16(B[0],TB[6]);
  B[12]=vaddq_u16(B[12],TB[6]);
  B[1]=vaddq_u16(B[1],TB[7]);
  B[13]=vaddq_u16(B[13],TB[7]);
  B[2]=vaddq_u16(B[2],TB[8]);
  B[14]=vaddq_u16(B[14],TB[8]);
  B[3]=vaddq_u16(B[3],TB[9]);
  B[15]=vaddq_u16(B[15],TB[9]);
  B[4]=vaddq_u16(B[4],TB[10]);
  B[16]=vaddq_u16(B[16],TB[10]);
  B[5]=vaddq_u16(B[5],TB[11]);
  B[17]=vaddq_u16(B[17],TB[11]);
  TB[12]=vld1q_u16(TmpB+8*60);
  TB[13]=vld1q_u16(TmpB+8*61);
  TB[14]=vld1q_u16(TmpB+8*62);
  TB[15]=vld1q_u16(TmpB+8*63);
  TB[16]=vld1q_u16(TmpB+8*64);
  TB[17]=vld1q_u16(TmpB+8*65);
  T[0]=vld1q_u16(TmpB+8*66);
  T[1]=vld1q_u16(TmpB+8*67);
  T[2]=vld1q_u16(TmpB+8*68);
  T[3]=vld1q_u16(TmpB+8*69);
  T[4]=vld1q_u16(TmpB+8*70);
  T[5]=vld1q_u16(TmpB+8*71);
  TB[12]=vaddq_u16(TB[12],T[0]);
  TB[12]=vaddq_u16(TB[12],T[2]);
  TB[14]=vaddq_u16(TB[14],T[0]);
  TB[14]=vaddq_u16(TB[14],T[4]);
  TB[16]=vaddq_u16(TB[16],T[2]);
  TB[16]=vaddq_u16(TB[16],T[4]);
  TB[13]=vaddq_u16(TB[13],T[1]);
  TB[13]=vaddq_u16(TB[13],T[3]);
  TB[15]=vaddq_u16(TB[15],T[1]);
  TB[15]=vaddq_u16(TB[15],T[5]);
  TB[17]=vaddq_u16(TB[17],T[3]);
  TB[17]=vaddq_u16(TB[17],T[5]);
  B[6]=vaddq_u16(B[6],TB[12]);
  B[12]=vaddq_u16(B[12],TB[12]);
  B[7]=vaddq_u16(B[7],TB[13]);
  B[13]=vaddq_u16(B[13],TB[13]);
  B[8]=vaddq_u16(B[8],TB[14]);
  B[14]=vaddq_u16(B[14],TB[14]);
  B[9]=vaddq_u16(B[9],TB[15]);
  B[15]=vaddq_u16(B[15],TB[15]);
  B[10]=vaddq_u16(B[10],TB[16]);
  B[16]=vaddq_u16(B[16],TB[16]);
  B[11]=vaddq_u16(B[11],TB[17]);
  B[17]=vaddq_u16(B[17],TB[17]);
  vst1q_u16(VecB+0,B[16]);
  vst1q_u16(VecB+8,B[17]);
  vst1q_u16(VecB+16,B[14]);
  vst1q_u16(VecB+24,B[15]);
  vst1q_u16(VecB+32,B[12]);
  vst1q_u16(VecB+40,B[13]);
  vst1q_u16(VecB+48,B[10]);
  vst1q_u16(VecB+56,B[11]);
  vst1q_u16(VecB+64,B[8]);
  vst1q_u16(VecB+72,B[9]);
  vst1q_u16(VecB+80,B[6]);
  vst1q_u16(VecB+88,B[7]);
  vst1q_u16(VecB+96,B[4]);
  vst1q_u16(VecB+104,B[5]);
  vst1q_u16(VecB+112,B[2]);
  vst1q_u16(VecB+120,B[3]);
  vst1q_u16(VecB+128,B[0]);
  vst1q_u16(VecB+136,B[1]);
}


/*
tmvp5332 method.
1 tmvp5, and 9 tmvp332.
The last 16 coefficients of the last tmvp332 is unused.
*/
void tmvp(uint16_t *restrict polyC, uint16_t *restrict polyA, uint16_t *restrict polyB){
    uint16_t tmp[SB0 * 9 * 3]; // SB0 = 144

    uint16_t *toepa = &tmp[0 * SB0]; /* nine 144*144 toeplitz matrix, needs nine length-288 vectors to store */
    uint16_t *kbw   = &tmp[18* SB0]; /* nine 144*144 vectors*/
    //uint16_t *kcw   = &tmp[27* SB0]; /* nine 144*144 vectors*/

    ittc5(toepa, polyA); /* 2.5k cycles */

    tc5(kbw, polyB); /* 0.6k cycles */


    for(int i = 0; i <= 8; i++){
      tmvp_144_ka33_ka2(&kbw[i * SB0], &toepa[i * SB0 * 2]);
    }

    ttc5(polyC, kbw); /* 0.8k cycles */

}


#if (NN==701)
#define SIZE 701

#define REPS 10

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void polymul_ref(uint16_t *C,  uint16_t *A, uint16_t *B) {
  int i, j;
  for (i=0; i<SIZE; i++) {
    C[i] = 0;
    for (j=0; j<=i; j++) C[i] += A[j] * B[i-j];
    for (j=i+1; j<SIZE; j++) C[i] += A[j] * B[SIZE+i-j];
  }
}

int main(){
  // mul 701 test
  int c1,c2,cc[REPS],c=0;
  int i,j;
  uint16_t C1[SIZE], C2[SIZE], A[SIZE], B[SIZE];
  
#ifndef __aarch64__
  hal_init_perfcounters(1,1);
#endif
  
  for (i=0; i<REPS; i++) {
    for (j=0; j<SIZE; j++) {
      A[j] = rand(); B[j] = rand();
    }
    c1 = hal_get_time();
    tmvp(C1,A,B);
    c2 = hal_get_time();
    c += (cc[i] = c2-c1);
    polymul_ref(C2,A,B);

    for (j=0; j<SIZE; j++) {
      if ((C1[j]-C2[j])&8191) {
	printf("%d %4x %4x\n",j,C1[j]%8191,C2[j]%8191);
	break;
      }
    }
  }
  qsort(cc, REPS, sizeof(int), cmpfunc);
  printf("everything okay, avg time = %d, median = %d\n",c/REPS,cc[REPS>>1]);
}

 #endif
