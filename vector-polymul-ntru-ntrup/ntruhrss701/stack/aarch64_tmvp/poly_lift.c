
#include <arm_neon.h>
#include "poly.h"

#include <assert.h>

void poly_lift(poly *r, poly *a)
{
  /* NOTE: Assumes input is in {0,1,2}^N */
  /*       Produces output in [0,Q-1]^N */
  poly b;
  uint16_t t;

  uint16x8_t t0, t1, t2;
  uint16x8_t acc0, acc1, acc2;

  uint16x8x3_t v30, v31, v32;

  /* Define z by <z*x^i, x-1> = delta_{i,0} mod 3:      */
  /*   t      = -1/N mod p = -N mod 3                   */
  /*   z[0]   = 2 - t mod 3                             */
  /*   z[1]   = 0 mod 3                                 */
  /*   z[j]   = z[j-1] + t mod 3                        */
  /* We'll compute b = a/(x-1) mod (3, Phi) using       */
  /*   b[0] = <z, a>, b[1] = <z*x,a>, b[2] = <z*x^2,a>  */
  /*   b[i] = b[i-3] - (a[i] + a[i-1] + a[i-2])         */
  t = 3 - (NTRU_N % 3);
  b.coeffs[0] = a->coeffs[0] * (2-t) + a->coeffs[1] * 0 + a->coeffs[2] * t;
  b.coeffs[1] = a->coeffs[1] * (2-t) + a->coeffs[2] * 0;
  b.coeffs[2] = a->coeffs[2] * (2-t);

  for(size_t i = NTRU_N; i < POLY_N; i++){
    a->coeffs[i] = 0;
  }

  for(size_t i = 3; i < 27; i++){
      b.coeffs[i] = a->coeffs[i];
  }


  for(size_t i = 27; i < NTRU_N - 2; i += 24){
      acc0 = vld1q_u16(&b.coeffs[ 3]);
      acc1 = vld1q_u16(&b.coeffs[11]);
      acc2 = vld1q_u16(&b.coeffs[19]);
      t0 = vld1q_u16(&a->coeffs[i + 8 * 0]);
      t1 = vld1q_u16(&a->coeffs[i + 8 * 1]);
      t2 = vld1q_u16(&a->coeffs[i + 8 * 2]);
      acc0 = acc0 + t0;
      acc1 = acc1 + t1;
      acc2 = acc2 + t2;
      vst1q_u16(&b.coeffs[ 3], acc0);
      vst1q_u16(&b.coeffs[11], acc1);
      vst1q_u16(&b.coeffs[19], acc2);
  }

  b.coeffs[3] += a->coeffs[NTRU_N - 2];
  b.coeffs[4] += a->coeffs[NTRU_N - 1];

  for(size_t i = 6; i < 27; i += 3){
      b.coeffs[3] += b.coeffs[i + 0];
      b.coeffs[4] += b.coeffs[i + 1];
      b.coeffs[5] += b.coeffs[i + 2];
  }


  b.coeffs[0] += b.coeffs[3] * 2;
  b.coeffs[1] += b.coeffs[3] * 1;
  b.coeffs[2] += b.coeffs[3] * 0;
  b.coeffs[0] += b.coeffs[4] * 3;
  b.coeffs[1] += b.coeffs[4] * 2;
  b.coeffs[2] += b.coeffs[4] * 1;
  b.coeffs[0] += b.coeffs[5] * 4;
  b.coeffs[1] += b.coeffs[5] * 3;
  b.coeffs[2] += b.coeffs[5] * 2;

  b.coeffs[1] += a->coeffs[0] * ((NTRU_N % 3) + t);
  b.coeffs[2] += a->coeffs[0] * (NTRU_N % 3);
  b.coeffs[2] += a->coeffs[1] * ((NTRU_N % 3) + t);

  for(size_t i = 3; i < NTRU_N - 24; i += 24)
    {
        v30 = vld3q_u16(&a->coeffs[i - 2]);
        v31 = vld3q_u16(&a->coeffs[i - 1]);
        v32 = vld3q_u16(&a->coeffs[i]);

        // 3k
        acc0 = v30.val[0] + v30.val[1];
        acc0 = acc0 + v30.val[2];
        acc0 = acc0 + acc0;

        // 3k
        b.coeffs[i + 3 * 0] = b.coeffs[i - 3 * 1] + vgetq_lane_u16(acc0, 0);
        b.coeffs[i + 3 * 1] = b.coeffs[i + 3 * 0] + vgetq_lane_u16(acc0, 1);
        b.coeffs[i + 3 * 2] = b.coeffs[i + 3 * 1] + vgetq_lane_u16(acc0, 2);
        b.coeffs[i + 3 * 3] = b.coeffs[i + 3 * 2] + vgetq_lane_u16(acc0, 3);
        b.coeffs[i + 3 * 4] = b.coeffs[i + 3 * 3] + vgetq_lane_u16(acc0, 4);
        b.coeffs[i + 3 * 5] = b.coeffs[i + 3 * 4] + vgetq_lane_u16(acc0, 5);
        b.coeffs[i + 3 * 6] = b.coeffs[i + 3 * 5] + vgetq_lane_u16(acc0, 6);
        b.coeffs[i + 3 * 7] = b.coeffs[i + 3 * 6] + vgetq_lane_u16(acc0, 7);

        // 3k+1
        acc1 = v31.val[0] + v31.val[1];
        acc1 = acc1 + v31.val[2];
        acc1 = acc1 + acc1;

        b.coeffs[1 + i + 3 * 0] = b.coeffs[1 + i - 3 * 1] + vgetq_lane_u16(acc1, 0);
        b.coeffs[1 + i + 3 * 1] = b.coeffs[1 + i + 3 * 0] + vgetq_lane_u16(acc1, 1);
        b.coeffs[1 + i + 3 * 2] = b.coeffs[1 + i + 3 * 1] + vgetq_lane_u16(acc1, 2);
        b.coeffs[1 + i + 3 * 3] = b.coeffs[1 + i + 3 * 2] + vgetq_lane_u16(acc1, 3);
        b.coeffs[1 + i + 3 * 4] = b.coeffs[1 + i + 3 * 3] + vgetq_lane_u16(acc1, 4);
        b.coeffs[1 + i + 3 * 5] = b.coeffs[1 + i + 3 * 4] + vgetq_lane_u16(acc1, 5);
        b.coeffs[1 + i + 3 * 6] = b.coeffs[1 + i + 3 * 5] + vgetq_lane_u16(acc1, 6);
        b.coeffs[1 + i + 3 * 7] = b.coeffs[1 + i + 3 * 6] + vgetq_lane_u16(acc1, 7);

        // 3k + 2
        acc2 = v32.val[0] + v32.val[1];
        acc2 = acc2 + v32.val[2];
        acc2 = acc2 + acc2;

        b.coeffs[2 + i + 3 * 0] = b.coeffs[2 + i - 3 * 1] + vgetq_lane_u16(acc2, 0);
        b.coeffs[2 + i + 3 * 1] = b.coeffs[2 + i + 3 * 0] + vgetq_lane_u16(acc2, 1);
        b.coeffs[2 + i + 3 * 2] = b.coeffs[2 + i + 3 * 1] + vgetq_lane_u16(acc2, 2);
        b.coeffs[2 + i + 3 * 3] = b.coeffs[2 + i + 3 * 2] + vgetq_lane_u16(acc2, 3);
        b.coeffs[2 + i + 3 * 4] = b.coeffs[2 + i + 3 * 3] + vgetq_lane_u16(acc2, 4);
        b.coeffs[2 + i + 3 * 5] = b.coeffs[2 + i + 3 * 4] + vgetq_lane_u16(acc2, 5);
        b.coeffs[2 + i + 3 * 6] = b.coeffs[2 + i + 3 * 5] + vgetq_lane_u16(acc2, 6);
        b.coeffs[2 + i + 3 * 7] = b.coeffs[2 + i + 3 * 6] + vgetq_lane_u16(acc2, 7);

    }
    b.coeffs[699] = b.coeffs[696] + 2*(a->coeffs[697] + a->coeffs[698] + a->coeffs[699]);
    b.coeffs[700] = b.coeffs[697] + 2*(a->coeffs[698] + a->coeffs[699] + a->coeffs[700]);

  /* Finish reduction mod Phi by subtracting Phi * b[N-1] */
  poly_mod_3_Phi_n(&b);

  /* Switch from {0,1,2} to {0,1,q-1} coefficient representation */
  poly_Z3_to_Zq(&b);

  /* Multiply by (x-1) */
  r->coeffs[0] = -(b.coeffs[0]);
  for(size_t i = 0; i < NTRU_N - 1; i++) {
    r->coeffs[i+1] = b.coeffs[i] - b.coeffs[i+1];
  }

}


