
#include <arm_neon.h>
#include "poly.h"

#include <stddef.h>

static uint8_t mod_tbl[64] = {
0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0
};

static inline uint16_t mod3(uint16_t a)
{
  uint16_t r;
  int16_t t, c;

  r = (a >> 8) + (a & 0xff); // r mod 255 == a mod 255
  r = (r >> 4) + (r & 0xf); // r' mod 15 == r mod 15
  r = (r >> 2) + (r & 0x3); // r' mod 3 == r mod 3
  r = (r >> 2) + (r & 0x3); // r' mod 3 == r mod 3

  t = r - 3;
  c = t >> 15;

  return (c&r) ^ (~c&t);
}

// static inline uint16x8_t mod3x8(uint16x8_t a){

//     uint16x8_t r;
//     uint16x8_t mask_0xff = vdupq_n_u16(0xff);
//     uint16x8_t mask_0x0f = vdupq_n_u16(0x0f);

//     uint8x16x4_t table = vld1q_u8_x4(mod_tbl);

//     r = vshrq_n_u16(a, 8) + vandq_u16(a, mask_0xff);
//     r = vshrq_n_u16(r, 4) + vandq_u16(r, mask_0x0f);
//     r = (uint16x8_t)vqtbl4q_u8(table, (uint8x16_t)r);

//     return r;

// }

void poly_mod_3_Phi_n(poly *r)
{


    uint16x8_t t;
    uint16x8_t mask_0xff = vdupq_n_u16(0xff);
    uint16x8_t mask_0x0f = vdupq_n_u16(0x0f);
    uint16x8_t last = vdupq_n_u16(mod3(2 * r->coeffs[NTRU_N - 1]));

    uint8x16x4_t table = vld1q_u8_x4(mod_tbl);

    for(size_t i = 0; i < POLY_N; i += 8){
        t = vld1q_u16(&r->coeffs[i]) + last;
        t = vshrq_n_u16(t, 8) + vandq_u16(t, mask_0xff);
        t = vshrq_n_u16(t, 4) + vandq_u16(t, mask_0x0f);
        t = (uint16x8_t)vqtbl4q_u8(table, (uint8x16_t)t);
        vst1q_u16(&r->coeffs[i], t);
    }

}

void poly_mod_q_Phi_n(poly *r)
{
  int i;
  for(i=0; i<NTRU_N; i++)
    r->coeffs[i] = r->coeffs[i] - r->coeffs[NTRU_N-1];
}

void poly_Rq_to_S3(poly *r, const poly *a)
{

    uint16_t last_uint16;
    uint16_t flag;

    uint16x8_t t;
    uint16x8_t mask_0xff = vdupq_n_u16(0xff);
    uint16x8_t mask_0x0f = vdupq_n_u16(0x0f);
    uint16x8_t mask_Q = vdupq_n_u16(NTRU_Q - 1);
    uint16x8_t last;

    uint8x16x4_t table = vld1q_u8_x4(mod_tbl);


    last_uint16 = r->coeffs[NTRU_N - 1] & (NTRU_Q - 1);
    flag = last_uint16 >> (NTRU_LOGQ - 1);
    last_uint16 += flag << (1 - (NTRU_LOGQ & 1));

    last = vdupq_n_u16(mod3(last_uint16 << 1));

    for(size_t i = 0; i < POLY_N; i += 8){
        t = vld1q_u16(&a->coeffs[i]);
        t = vandq_u16(t, mask_Q);
        t += vshlq_n_u16(vshrq_n_u16(t, NTRU_LOGQ - 1), (1 - (NTRU_LOGQ & 1)));
        t += last;
        t = vshrq_n_u16(t, 8) + vandq_u16(t, mask_0xff);
        t = vshrq_n_u16(t, 4) + vandq_u16(t, mask_0x0f);
        t = (uint16x8_t)vqtbl4q_u8(table, (uint8x16_t)t);
        vst1q_u16(&r->coeffs[i], t);
    }
}

