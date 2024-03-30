
#include <arm_neon.h>
#include "batch_multiplication.h"


#define SB_HALF 28 // Round up of (9*5*5-1)/8

void schoolbook_8x8(uint16_t *restrict c_in_mem,
                         uint16_t *restrict a_in_mem,
                         uint16_t *restrict b_in_mem) {
    uint16x8_t tmp, aa[8], bb[8], zero;
    zero = vdupq_n_u16(0);
    uint16_t *a_mem = a_in_mem, *b_mem = b_in_mem, *c_mem = c_in_mem;
    for (uint16_t i = 0; i < 84; i++) {
        aa[0] = vld1q_u16(&a_mem[0 * 8]);
        bb[0] = vld1q_u16(&b_mem[0 * 8]);
        aa[1] = vld1q_u16(&a_mem[1 * 8]);
        bb[1] = vld1q_u16(&b_mem[1 * 8]);
        aa[2] = vld1q_u16(&a_mem[2 * 8]);
        bb[2] = vld1q_u16(&b_mem[2 * 8]);
        aa[3] = vld1q_u16(&a_mem[3 * 8]);
        bb[3] = vld1q_u16(&b_mem[3 * 8]);
        aa[4] = vld1q_u16(&a_mem[4 * 8]);
        bb[4] = vld1q_u16(&b_mem[4 * 8]);
        aa[5] = vld1q_u16(&a_mem[5 * 8]);
        bb[5] = vld1q_u16(&b_mem[5 * 8]);
        aa[6] = vld1q_u16(&a_mem[6 * 8]);
        bb[6] = vld1q_u16(&b_mem[6 * 8]);
        aa[7] = vld1q_u16(&a_mem[7 * 8]);
        bb[7] = vld1q_u16(&b_mem[7 * 8]);

        uint16x8_t y0, y1, y2, y3, y4, y5, y6, y7, 
                   y8, y9, y10, y11, y12, y13, y14,
                   y15;
        y0 = aa[0];
        y1 = aa[1];
        y2 = aa[2];
        y3 = aa[3];
        y4 = aa[4];
        y5 = aa[5];
        y6 = aa[6];
        y7 = aa[7];

    // Transpose 8x8
    y8 = vtrn1q_u16(y0, y1);
    y9 = vtrn2q_u16(y0, y1);
    y10 = vtrn1q_u16(y2, y3);
    y11 = vtrn2q_u16(y2, y3);
    y12 = vtrn1q_u16(y4, y5);
    y13 = vtrn2q_u16(y4, y5);
    y14 = vtrn1q_u16(y6, y7);
    y15 = vtrn2q_u16(y6, y7);

    y0 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y1 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y2 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y3 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y4 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y5 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y6 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y13, (uint32x4_t)y15);
    y7 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y13, (uint32x4_t)y15);

    y8  = (uint16x8_t)vtrn1q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y9  = (uint16x8_t)vtrn2q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y10 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y11 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y12 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y13 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y14 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y3, (uint64x2_t)y7);
    y15 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y3, (uint64x2_t)y7);

        aa[0] = y8;
        aa[1] = y12;
        aa[2] = y10;
        aa[3] = y14;
        aa[4] = y9;
        aa[5] = y13;
        aa[6] = y11;
        aa[7] = y15;

        y0 = bb[0];
        y1 = bb[1];
        y2 = bb[2];
        y3 = bb[3];
        y4 = bb[4];
        y5 = bb[5];
        y6 = bb[6];
        y7 = bb[7];

    // Transpose 8x8
    y8 = vtrn1q_u16(y0, y1);
    y9 = vtrn2q_u16(y0, y1);
    y10 = vtrn1q_u16(y2, y3);
    y11 = vtrn2q_u16(y2, y3);
    y12 = vtrn1q_u16(y4, y5);
    y13 = vtrn2q_u16(y4, y5);
    y14 = vtrn1q_u16(y6, y7);
    y15 = vtrn2q_u16(y6, y7);

    y0 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y1 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y2 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y3 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y4 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y5 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y6 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y13, (uint32x4_t)y15);
    y7 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y13, (uint32x4_t)y15);

    y8  = (uint16x8_t)vtrn1q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y9  = (uint16x8_t)vtrn2q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y10 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y11 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y12 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y13 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y14 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y3, (uint64x2_t)y7);
    y15 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y3, (uint64x2_t)y7);

        bb[0] = y8;
        bb[1] = y12;
        bb[2] = y10;
        bb[3] = y14;
        bb[4] = y9;
        bb[5] = y13;
        bb[6] = y11;
        bb[7] = y15;



        tmp = vmulq_u16(aa[0], bb[0]);
        y0 = tmp;
        //vst1q_u16(&c_mem[16 * 0], tmp);
        //----

        tmp = vmulq_u16(aa[0], bb[1]);
        tmp = vmlaq_u16(tmp, aa[1], bb[0]);
        y1 = tmp;
        //vst1q_u16(&c_mem[16 * 1], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[2]);
        tmp = vmlaq_u16(tmp, aa[1], bb[1]);
        tmp = vmlaq_u16(tmp, aa[2], bb[0]);
        y2 = tmp;
        //vst1q_u16(&c_mem[16 * 2], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[3]);
        tmp = vmlaq_u16(tmp, aa[1], bb[2]);
        tmp = vmlaq_u16(tmp, aa[2], bb[1]);
        tmp = vmlaq_u16(tmp, aa[3], bb[0]);
        y3 = tmp;
        //vst1q_u16(&c_mem[16 * 3], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[4]);
        tmp = vmlaq_u16(tmp, aa[1], bb[3]);
        tmp = vmlaq_u16(tmp, aa[2], bb[2]);
        tmp = vmlaq_u16(tmp, aa[3], bb[1]);
        tmp = vmlaq_u16(tmp, aa[4], bb[0]);
        y4 = tmp;
        //vst1q_u16(&c_mem[16 * 4], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[5]);
        tmp = vmlaq_u16(tmp, aa[1], bb[4]);
        tmp = vmlaq_u16(tmp, aa[2], bb[3]);
        tmp = vmlaq_u16(tmp, aa[3], bb[2]);
        tmp = vmlaq_u16(tmp, aa[4], bb[1]);
        tmp = vmlaq_u16(tmp, aa[5], bb[0]);
        y5 = tmp;
        //vst1q_u16(&c_mem[16 * 5], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[6]);
        tmp = vmlaq_u16(tmp, aa[1], bb[5]);
        tmp = vmlaq_u16(tmp, aa[2], bb[4]);
        tmp = vmlaq_u16(tmp, aa[3], bb[3]);
        tmp = vmlaq_u16(tmp, aa[4], bb[2]);
        tmp = vmlaq_u16(tmp, aa[5], bb[1]);
        tmp = vmlaq_u16(tmp, aa[6], bb[0]);
        y6 = tmp;
        //vst1q_u16(&c_mem[16 * 6], tmp);
        //----
        
        tmp = vmulq_u16(aa[0], bb[7]);
        tmp = vmlaq_u16(tmp, aa[1], bb[6]);
        tmp = vmlaq_u16(tmp, aa[2], bb[5]);
        tmp = vmlaq_u16(tmp, aa[3], bb[4]);
        tmp = vmlaq_u16(tmp, aa[4], bb[3]);
        tmp = vmlaq_u16(tmp, aa[5], bb[2]);
        tmp = vmlaq_u16(tmp, aa[6], bb[1]);
        tmp = vmlaq_u16(tmp, aa[7], bb[0]);
        y7 = tmp;
        //vst1q_u16(&c_mem[16 * 7], tmp);

    // Transpose 8x8
    y8 = vtrn1q_u16(y0, y1);
    y9 = vtrn2q_u16(y0, y1);
    y10 = vtrn1q_u16(y2, y3);
    y11 = vtrn2q_u16(y2, y3);
    y12 = vtrn1q_u16(y4, y5);
    y13 = vtrn2q_u16(y4, y5);
    y14 = vtrn1q_u16(y6, y7);
    y15 = vtrn2q_u16(y6, y7);

    y0 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y1 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y2 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y3 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y4 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y5 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y6 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y13, (uint32x4_t)y15);
    y7 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y13, (uint32x4_t)y15);

    y8  = (uint16x8_t)vtrn1q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y9  = (uint16x8_t)vtrn2q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y10 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y11 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y12 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y13 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y14 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y3, (uint64x2_t)y7);
    y15 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y3, (uint64x2_t)y7);
    // 16x16: STR A1
    vst1q_u16(c_mem + 16*0, y8);
    vst1q_u16(c_mem + 16*1, y12);
    vst1q_u16(c_mem + 16*2, y10);
    vst1q_u16(c_mem + 16*3, y14);
    vst1q_u16(c_mem + 16*4, y9);
    vst1q_u16(c_mem + 16*5, y13);
    vst1q_u16(c_mem + 16*6, y11);
    vst1q_u16(c_mem + 16*7, y15);
      
        // ----------------PART 2----------------
        tmp = vmulq_u16(aa[1], bb[7]);
        tmp = vmlaq_u16(tmp, aa[2], bb[6]);
        tmp = vmlaq_u16(tmp, aa[3], bb[5]);
        tmp = vmlaq_u16(tmp, aa[4], bb[4]);
        tmp = vmlaq_u16(tmp, aa[5], bb[3]);
        tmp = vmlaq_u16(tmp, aa[6], bb[2]);
        tmp = vmlaq_u16(tmp, aa[7], bb[1]);
        y0 = tmp;
        //vst1q_u16(&c_mem[16 * 0 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[2], bb[7]);
        tmp = vmlaq_u16(tmp, aa[3], bb[6]);
        tmp = vmlaq_u16(tmp, aa[4], bb[5]);
        tmp = vmlaq_u16(tmp, aa[5], bb[4]);
        tmp = vmlaq_u16(tmp, aa[6], bb[3]);
        tmp = vmlaq_u16(tmp, aa[7], bb[2]);
        y1 = tmp;
        //vst1q_u16(&c_mem[16 * 1 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[3], bb[7]);
        tmp = vmlaq_u16(tmp, aa[4], bb[6]);
        tmp = vmlaq_u16(tmp, aa[5], bb[5]);
        tmp = vmlaq_u16(tmp, aa[6], bb[4]);
        tmp = vmlaq_u16(tmp, aa[7], bb[3]);
        y2 = tmp;
        //vst1q_u16(&c_mem[16 * 2 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[4], bb[7]);
        tmp = vmlaq_u16(tmp, aa[5], bb[6]);
        tmp = vmlaq_u16(tmp, aa[6], bb[5]);
        tmp = vmlaq_u16(tmp, aa[7], bb[4]);
        y3 = tmp;
        //vst1q_u16(&c_mem[16 * 3 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[5], bb[7]);
        tmp = vmlaq_u16(tmp, aa[6], bb[6]);
        tmp = vmlaq_u16(tmp, aa[7], bb[5]);
        y4 = tmp;
        //vst1q_u16(&c_mem[16 * 4 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[6], bb[7]);
        tmp = vmlaq_u16(tmp, aa[7], bb[6]);
        y5 = tmp;
        //vst1q_u16(&c_mem[16 * 5 + 8], tmp);
        //-----
        tmp = vmulq_u16(aa[7], bb[7]);
        y6 = tmp;
        //vst1q_u16(&c_mem[16 * 6 + 8], tmp);
        //-----
        y7 = zero;
        //vst1q_u16(&c_mem[16 * 7 + 8], zero);

    // Transpose 8x8
    y8 = vtrn1q_u16(y0, y1);
    y9 = vtrn2q_u16(y0, y1);
    y10 = vtrn1q_u16(y2, y3);
    y11 = vtrn2q_u16(y2, y3);
    y12 = vtrn1q_u16(y4, y5);
    y13 = vtrn2q_u16(y4, y5);
    y14 = vtrn1q_u16(y6, y7);
    y15 = vtrn2q_u16(y6, y7);

    y0 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y1 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y8, (uint32x4_t)y10);
    y2 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y3 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y9, (uint32x4_t)y11);
    y4 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y5 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y12, (uint32x4_t)y14);
    y6 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y13, (uint32x4_t)y15);
    y7 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y13, (uint32x4_t)y15);

    y8  = (uint16x8_t)vtrn1q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y9  = (uint16x8_t)vtrn2q_u64((uint64x2_t)y0, (uint64x2_t)y4);
    y10 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y11 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y1, (uint64x2_t)y5);
    y12 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y13 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y2, (uint64x2_t)y6);
    y14 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y3, (uint64x2_t)y7);
    y15 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y3, (uint64x2_t)y7);

    // 16x16: STR A2<-A2
    vst1q_u16(c_mem + 16*0 + 8, y8);
    vst1q_u16(c_mem + 16*1 + 8, y12);
    vst1q_u16(c_mem + 16*2 + 8, y10);
    vst1q_u16(c_mem + 16*3 + 8, y14);
    vst1q_u16(c_mem + 16*4 + 8, y9);
    vst1q_u16(c_mem + 16*5 + 8, y13);
    vst1q_u16(c_mem + 16*6 + 8, y11);
    vst1q_u16(c_mem + 16*7 + 8, y15);

        a_mem += 64;
        b_mem += 64;
        c_mem += 128;
    }
}




