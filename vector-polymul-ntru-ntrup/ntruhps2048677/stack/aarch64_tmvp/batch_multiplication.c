#include <arm_neon.h>
#include "batch_multiplication.h"

#define ITER 25
/*
c_in_mem is the output length-16 vector,
a_in_mem is the 8x8 Toeplitz matrix, A, {1st - 8th} contains the reverse of first row of A, {8th - 15th} contains the first column of A
b_in_mem is the input length-16 vector,
*/
void tmvp2_8x8(uint16_t *restrict c_in_mem, uint16_t *restrict a_in_mem)
{
    uint16x8_t r0, r1, r2, r3, r4, r5, r6, r7, b0, b1, b2;
    uint16x8_t p2, p1, p0, c0, c1;
    uint16_t *a_mem = a_in_mem, *b_mem, *c_mem = c_in_mem;
    b_mem = c_mem;

    for (uint16_t addr = 0; addr < ITER*16; addr+=16)
    {
//tmvp2 split vector, results are in b0, b1, b2
        b0 = vld1q_u16(&b_mem[0+addr]);
        b1 = vld1q_u16(&b_mem[8+addr]);
        b2 = vaddq_u16(b0, b1);

//3x 8x8 toeplitz schoolbook, results are in p2, p1, p0
        r0 = vld1q_u16(&a_mem[8+addr*3]);
        r1 = vld1q_u16(&a_mem[7+addr*3]);
        r2 = vld1q_u16(&a_mem[6+addr*3]);
        r3 = vld1q_u16(&a_mem[5+addr*3]);
        r4 = vld1q_u16(&a_mem[4+addr*3]);
        r5 = vld1q_u16(&a_mem[3+addr*3]);
        r6 = vld1q_u16(&a_mem[2+addr*3]);
        r7 = vld1q_u16(&a_mem[1+addr*3]);

        p2 = vmulq_laneq_u16(r0, b0, 0);
        p2 = vmlaq_laneq_u16(p2, r1, b0, 1);
        p2 = vmlaq_laneq_u16(p2, r2, b0, 2);
        p2 = vmlaq_laneq_u16(p2, r3, b0, 3);
        p2 = vmlaq_laneq_u16(p2, r4, b0, 4);
        p2 = vmlaq_laneq_u16(p2, r5, b0, 5);
        p2 = vmlaq_laneq_u16(p2, r6, b0, 6);
        p2 = vmlaq_laneq_u16(p2, r7, b0, 7);


        r0 = vld1q_u16(&a_mem[16+8+addr*3]);
        r1 = vld1q_u16(&a_mem[16+7+addr*3]);
        r2 = vld1q_u16(&a_mem[16+6+addr*3]);
        r3 = vld1q_u16(&a_mem[16+5+addr*3]);
        r4 = vld1q_u16(&a_mem[16+4+addr*3]);
        r5 = vld1q_u16(&a_mem[16+3+addr*3]);
        r6 = vld1q_u16(&a_mem[16+2+addr*3]);
        r7 = vld1q_u16(&a_mem[16+1+addr*3]);

        p1 = vmulq_laneq_u16(r0, b1, 0);
        p1 = vmlaq_laneq_u16(p1, r1, b1, 1);
        p1 = vmlaq_laneq_u16(p1, r2, b1, 2);
        p1 = vmlaq_laneq_u16(p1, r3, b1, 3);
        p1 = vmlaq_laneq_u16(p1, r4, b1, 4);
        p1 = vmlaq_laneq_u16(p1, r5, b1, 5);
        p1 = vmlaq_laneq_u16(p1, r6, b1, 6);
        p1 = vmlaq_laneq_u16(p1, r7, b1, 7);


        r0 = vld1q_u16(&a_mem[32+8+addr*3]);
        r1 = vld1q_u16(&a_mem[32+7+addr*3]);
        r2 = vld1q_u16(&a_mem[32+6+addr*3]);
        r3 = vld1q_u16(&a_mem[32+5+addr*3]);
        r4 = vld1q_u16(&a_mem[32+4+addr*3]);
        r5 = vld1q_u16(&a_mem[32+3+addr*3]);
        r6 = vld1q_u16(&a_mem[32+2+addr*3]);
        r7 = vld1q_u16(&a_mem[32+1+addr*3]);

        p0 = vmulq_laneq_u16(r0, b2, 0);
        p0 = vmlaq_laneq_u16(p0, r1, b2, 1);
        p0 = vmlaq_laneq_u16(p0, r2, b2, 2);
        p0 = vmlaq_laneq_u16(p0, r3, b2, 3);
        p0 = vmlaq_laneq_u16(p0, r4, b2, 4);
        p0 = vmlaq_laneq_u16(p0, r5, b2, 5);
        p0 = vmlaq_laneq_u16(p0, r6, b2, 6);
        p0 = vmlaq_laneq_u16(p0, r7, b2, 7);


//tmvp2 combine vector
        c0 = vaddq_u16(p0, p1);
        c1 = vsubq_u16(p0, p2);

        vst1q_u16(&c_mem[0+addr], c0);
        vst1q_u16(&c_mem[8+addr], c1);
    }
}
