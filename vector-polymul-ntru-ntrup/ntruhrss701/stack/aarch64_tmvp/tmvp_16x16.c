#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "cpucycles.h"

#define REPS 10
void tmvp_144_ka33_ka2(uint16_t *VecB, uint16_t *restrict ToepA);

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void tmvp_ref(uint16_t  *VecB, uint16_t  *restrict ToepA, const int n){
  uint16_t  tmpC[n];
  int i, j;
  
  for (i=0; i<n; i++) {
    tmpC[i] = 0;
    for (j=0; j<n; j++) {
      tmpC[i] += ToepA[n+i-j] * VecB[j];
    }  
  }
  for (i=0; i<n; i++) VecB[i] = tmpC[i];
}

void tmvp_16x16_ref(uint16_t  *VecB, uint16_t  *restrict ToepA){
  tmvp_ref(VecB, ToepA, 16);
}
void tmvp_8x8_ref(uint16_t  *VecB, uint16_t  *restrict ToepA){
  tmvp_ref(VecB, ToepA, 8);
}
void tmvp_48x48_ref(uint16_t  *VecB, uint16_t  *restrict ToepA){
  tmvp_ref(VecB, ToepA, 48);
}

void tmvp_8x8(uint16_t  *VecB, uint16_t  *restrict ToepA){
  uint16_t  tmpC[8];
  int i, j;
  uint16x8_t A0, A1, B, T, TC;
  A0 = vld1q_u16(ToepA);
  A1 = vld1q_u16(ToepA+8);
  B = vld1q_u16(VecB);
  TC = vmulq_laneq_u16(A1, B, 0);
  T = vextq_u16(A0, A1, 7);
  TC = vmlaq_laneq_u16(TC, T, B, 1);
  T = vextq_u16(A0, A1, 6);
  TC = vmlaq_laneq_u16(TC, T, B, 2);
  T = vextq_u16(A0, A1, 5);
  TC = vmlaq_laneq_u16(TC, T, B, 3);
  T = vextq_u16(A0, A1, 4);
  TC = vmlaq_laneq_u16(TC, T, B, 4);
  T = vextq_u16(A0, A1, 3);
  TC = vmlaq_laneq_u16(TC, T, B, 5);
  T = vextq_u16(A0, A1, 2);
  TC = vmlaq_laneq_u16(TC, T, B, 6);
  T = vextq_u16(A0, A1, 1);
  TC = vmlaq_laneq_u16(TC, T, B, 7);
  vst1q_u16(VecB,TC);
}



void tmvp_16x16_ka(uint16_t  *VecB, uint16_t  *restrict ToepA) {

  uint16x8_t A0, A1, A2, A3, B0, B1;
  uint16x8_t TA0, TA1, TA2, TB, TC0, TC1, TC2, T0, T1, T2;

  A0 = vld1q_u16(ToepA)   ; A1 = vld1q_u16(ToepA+8);
  A2 = vld1q_u16(ToepA+16); A3 = vld1q_u16(ToepA+24);
  B0 = vld1q_u16(VecB)    ; B1 = vld1q_u16(VecB+8);
  TA2 = vsubq_u16(A3,A2)  ; TA1 = vsubq_u16(A2,A1);
  TA0 = vsubq_u16(A1,A0)  ; TB = vaddq_u16(B0,B1);

  // compute TC0 = (TA1, TA2) x B0
  // compute TC1 = (TA0, TA1) x B1
  // compute TC2 = (A1, A2) x TB
 
  TC0 = vmulq_laneq_u16(TA2, B0, 0);
  TC1 = vmulq_laneq_u16(TA1, B1, 0);
  TC2 = vmulq_laneq_u16(A2, TB, 0);
  
  T0 = vextq_u16(TA1, TA2, 7);
  T1 = vextq_u16(TA0, TA1, 7);
  T2 = vextq_u16(A1, A2, 7);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 1);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 1);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 1);
  T0 = vextq_u16(TA1, TA2, 6);
  T1 = vextq_u16(TA0, TA1, 6);
  T2 = vextq_u16(A1, A2, 6);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 2);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 2);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 2);
  T0 = vextq_u16(TA1, TA2, 5);
  T1 = vextq_u16(TA0, TA1, 5);
  T2 = vextq_u16(A1, A2, 5);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 3);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 3);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 3);
  T0 = vextq_u16(TA1, TA2, 4);
  T1 = vextq_u16(TA0, TA1, 4);
  T2 = vextq_u16(A1, A2, 4);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 4);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 4);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 4);
  T0 = vextq_u16(TA1, TA2, 3);
  T1 = vextq_u16(TA0, TA1, 3);
  T2 = vextq_u16(A1, A2, 3);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 5);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 5);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 5);
  T0 = vextq_u16(TA1, TA2, 2);
  T1 = vextq_u16(TA0, TA1, 2);
  T2 = vextq_u16(A1, A2, 2);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 6);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 6);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 6);
  T0 = vextq_u16(TA1, TA2, 1);
  T1 = vextq_u16(TA0, TA1, 1);
  T2 = vextq_u16(A1, A2, 1);
  TC0 = vmlaq_laneq_u16(TC0, T0, B0, 7);
  TC1 = vmlaq_laneq_u16(TC1, T1, B1, 7);
  TC2 = vmlaq_laneq_u16(TC2, T2, TB, 7);

  T0 = vsubq_u16(TC2, TC1);
  T1 = vaddq_u16(TC2, TC0);

  vst1q_u16(VecB,T0);
  vst1q_u16(VecB+8,T1);
}
void tmvp_16x16_x2_ka(uint16_t  *VecB, uint16_t  *restrict ToepA, uint16_t  *vecb, uint16_t  *restrict toepa) {

  uint16x8_t A0, A1, A2, A3, B0, B1;
  uint16x8_t TA0, TA1, TA2, TB, TC0, TC1, TC2, T0, T1, T2;
  uint16x8_t a0, a1, a2, a3, b0, b1;
  uint16x8_t ta0, ta1, ta2, tb, tc0, tc1, tc2, t0, t1, t2;

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

void tmvp_48x48_ka3_ka2(uint16_t  *VecB, uint16_t *restrict ToepA){
  uint16_t TmpB[48], TmpA[96], *Y;
  uint16x8_t A[12], TA[12], B[6], TB[6], *X;
  int i;

  A[0] = vld1q_u16(ToepA + 8*0);
  A[1] = vld1q_u16(ToepA + 8*1);
  A[2] = vld1q_u16(ToepA + 8*2);
  A[3] = vld1q_u16(ToepA + 8*3);
  A[4] = vld1q_u16(ToepA + 8*4);
  A[5] = vld1q_u16(ToepA + 8*5);
  A[6] = vld1q_u16(ToepA + 8*6);
  A[7] = vld1q_u16(ToepA + 8*7);
  A[8] = vld1q_u16(ToepA + 8*8);
  A[9] = vld1q_u16(ToepA + 8*9);
  A[10] = vld1q_u16(ToepA + 8*10);
  A[11] = vld1q_u16(ToepA + 8*11);
  B[0] = vld1q_u16(VecB + 8*0);
  B[1] = vld1q_u16(VecB + 8*1);
  B[2] = vld1q_u16(VecB + 8*2);
  B[3] = vld1q_u16(VecB + 8*3);
  B[4] = vld1q_u16(VecB + 8*4);
  B[5] = vld1q_u16(VecB + 8*5);

  TA[0] = vsubq_u16(A[0],A[2]);
  TA[4] = vsubq_u16(A[4],A[2]);
  TA[8] = vsubq_u16(A[8],A[6]);
  TA[0] = vsubq_u16(TA[0],A[4]);
  TA[4] = vsubq_u16(TA[4],A[6]);
  TA[8] = vsubq_u16(TA[8],A[4]);
  TA[1] = vsubq_u16(A[1],A[3]);
  TA[5] = vsubq_u16(A[5],A[3]);
  TA[9] = vsubq_u16(A[9],A[7]);
  TA[1] = vsubq_u16(TA[1],A[5]);
  TA[5] = vsubq_u16(TA[5],A[7]);
  TA[9] = vsubq_u16(TA[9],A[5]);
  TA[2] = vsubq_u16(A[2],A[4]);
  TA[6] = vsubq_u16(A[6],A[4]);
  TA[10] = vsubq_u16(A[10],A[8]);
  TA[2] = vsubq_u16(TA[2],A[6]);
  TA[6] = vsubq_u16(TA[6],A[8]);
  TA[10] = vsubq_u16(TA[10],A[6]);
  TA[3] = vsubq_u16(A[3],A[5]);
  TA[7] = vsubq_u16(A[7],A[5]);
  TA[11] = vsubq_u16(A[11],A[9]);
  TA[3] = vsubq_u16(TA[3],A[7]);
  TA[7] = vsubq_u16(TA[7],A[9]);
  TA[11] = vsubq_u16(TA[11],A[7]);

  TB[0] = vaddq_u16(B[0],B[2]);
  TB[2] = vaddq_u16(B[0],B[4]);
  TB[4] = vaddq_u16(B[2],B[4]);
  TB[1] = vaddq_u16(B[1],B[3]);
  TB[3] = vaddq_u16(B[1],B[5]);
  TB[5] = vaddq_u16(B[3],B[5]);
  vst1q_u16(TmpA + 8*0, TA[0]);
  vst1q_u16(TmpA + 8*1, TA[1]);
  vst1q_u16(TmpA + 8*2, TA[2]);
  vst1q_u16(TmpA + 8*3, TA[3]);
  vst1q_u16(TmpA + 8*4, TA[4]);
  vst1q_u16(TmpA + 8*5, TA[5]);
  vst1q_u16(TmpA + 8*6, TA[6]);
  vst1q_u16(TmpA + 8*7, TA[7]);
  vst1q_u16(TmpA + 8*8, TA[8]);
  vst1q_u16(TmpA + 8*9, TA[9]);
  vst1q_u16(TmpA + 8*10, TA[10]);
  vst1q_u16(TmpA + 8*11, TA[11]);
  vst1q_u16(TmpB + 8*0, TB[0]);
  vst1q_u16(TmpB + 8*1, TB[1]);
  vst1q_u16(TmpB + 8*2, TB[2]);
  vst1q_u16(TmpB + 8*3, TB[3]);
  vst1q_u16(TmpB + 8*4, TB[4]);
  vst1q_u16(TmpB + 8*5, TB[5]);
 
tmvp_16x16_entry:
  tmvp_16x16_ka(VecB+16*0,TmpA+32*2);
  tmvp_16x16_ka(TmpB+16*0,ToepA+16*3);
  tmvp_16x16_ka(VecB+16*1,TmpA+32*1);
  tmvp_16x16_ka(TmpB+16*1,ToepA+16*2);
  tmvp_16x16_ka(VecB+16*2,TmpA+32*0);
  tmvp_16x16_ka(TmpB+16*2,ToepA+16*1);

tmvp_16x16_end:

  B[0] = vld1q_u16(VecB + 8*0);
  B[1] = vld1q_u16(VecB + 8*1);
  B[2] = vld1q_u16(VecB + 8*2);
  B[3] = vld1q_u16(VecB + 8*3);
  B[4] = vld1q_u16(VecB + 8*4);
  B[5] = vld1q_u16(VecB + 8*5);
  TB[0] = vld1q_u16(TmpB + 8*0);
  TB[1] = vld1q_u16(TmpB + 8*1);
  TB[2] = vld1q_u16(TmpB + 8*2);
  TB[3] = vld1q_u16(TmpB + 8*3);
  TB[4] = vld1q_u16(TmpB + 8*4);
  TB[5] = vld1q_u16(TmpB + 8*5);

  B[0] = vaddq_u16(B[0],TB[0]);
  B[0] = vaddq_u16(B[0],TB[2]);
  B[2] = vaddq_u16(B[2],TB[0]);
  B[2] = vaddq_u16(B[2],TB[4]);
  B[4] = vaddq_u16(B[4],TB[2]);
  B[4] = vaddq_u16(B[4],TB[4]);
  B[1] = vaddq_u16(B[1],TB[1]);
  B[1] = vaddq_u16(B[1],TB[3]);
  B[3] = vaddq_u16(B[3],TB[1]);
  B[3] = vaddq_u16(B[3],TB[5]);
  B[5] = vaddq_u16(B[5],TB[3]);
  B[5] = vaddq_u16(B[5],TB[5]);
  
  vst1q_u16(VecB+8*0,B[4]);
  vst1q_u16(VecB+8*1,B[5]);
  vst1q_u16(VecB+8*2,B[2]);
  vst1q_u16(VecB+8*3,B[3]);
  vst1q_u16(VecB+8*4,B[0]);
  vst1q_u16(VecB+8*5,B[1]);
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
  uint16_t TmpB[576], TmpA[1152], *Y;
  uint16x8_t A[36], TA[36], B[18], TB[18], T[6], *X;
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

#if (NN==144)
#define SIZE 144
int main(){
  // 48x48_ka3_ka2 test
  int c1,c2,cc[REPS],c=0;
  int i,j;
  uint16_t b1[SIZE], b2[SIZE], A[2*SIZE];
  
#ifndef __aarch64__
  hal_init_perfcounters(1,1);
#endif
  
  for (i=0; i<REPS; i++) {
    for (j=0; j<SIZE; j++) {
      b2[j] = b1[j] = rand();
      A[j*2] = rand(); A[j*2+1] = rand();
    }
    c1 = hal_get_time();
    tmvp_144_ka33_ka2(b1,A);
    c2 = hal_get_time();
    c += (cc[i] = c2-c1);
    tmvp_ref(b2,A,144);

    for (j=0; j<SIZE; j++) {
      if (b1[j]!=b2[j]) {
	printf("%d %ud %ud\n",j,b1[j],b2[j]);
	break;
      }
    }
  }
  qsort(cc, REPS, sizeof(int), cmpfunc);
  printf("everything okay, avg time = %d, median = %d\n",c/REPS,cc[REPS>>1]);
}

#elif (NN==48)
#define SIZE 48
int main(){
  // 48x48_ka3_ka2 test
  int c1,c2,cc[REPS],c=0;
  int i,j;
  uint16_t b1[SIZE], b2[SIZE], A[2*SIZE];
  
#ifndef __aarch64__
  hal_init_perfcounters(1,1);
#endif
  
  for (i=0; i<REPS; i++) {
    for (j=0; j<SIZE; j++) {
      b2[j] = b1[j] = rand();
      A[j*2] = rand(); A[j*2+1] = rand();
    }
    c1 = hal_get_time();
    tmvp_48x48_ka3_ka2(b1,A);
    c2 = hal_get_time();
    c += (cc[i] = c2-c1);
    tmvp_48x48_ref(b2,A);

    for (j=0; j<SIZE; j++) {
      if (b1[j]!=b2[j]) {
	printf("%d %ud %ud\n",j,b1[j],b2[j]);
	break;
      }
    }
  }
  qsort(cc, REPS, sizeof(int), cmpfunc);
  printf("everything okay, avg time = %d, median = %d\n",c/REPS,cc[REPS>>1]);
}


#elif (NN==16)
#define SIZE 16
int main(){
  // 16x16_x2_ka test
  int c1,c2,cc[REPS],c=0;
  int i,j;
  uint16_t b1[SIZE], b2[SIZE], A[2*SIZE];
  
#ifndef __aarch64__
  hal_init_perfcounters(1,1);
#endif
  
  for (i=0; i<REPS; i++) {
    for (j=0; j<SIZE; j++) {
      b2[j] = b1[j] = rand();
      A[j*2] = rand(); A[j*2+1] = rand();
    }
    c1 = hal_get_time();
    tmvp_16x16_ka(b1,A);
    c2 = hal_get_time();
    c += (cc[i] = c2-c1);
    tmvp_16x16_ref(b2,A);

    for (j=0; j<SIZE; j++) {
      if (b1[j]!=b2[j]) {
	printf("%d %ud %ud\n",j,b1[j],b2[j]);
	break;
      }
    }
  }
  qsort(cc, REPS, sizeof(int), cmpfunc);
  printf("everything okay, avg time = %d, median = %d\n",c/REPS,cc[REPS>>1]);
}

#endif


