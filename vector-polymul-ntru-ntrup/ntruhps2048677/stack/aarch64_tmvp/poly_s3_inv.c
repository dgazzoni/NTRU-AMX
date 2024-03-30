/* Based on supercop-20200702/crypto_core/invhrss701/simpler/core.c */

#include "poly.h"

#include <arm_neon.h>

#include <stdio.h>

static inline uint8_t mod3(uint8_t a) { /* a between 0 and 9 */
    int16_t t, c;
    a = (a >> 2) + (a & 3); /* between 0 and 4 */
    t = a - 3;
    c = t >> 5;
    return (uint8_t) (t ^ (c & (a ^ t)));
}

/* return -1 if x<0 and y<0; otherwise return 0 */
static inline int16_t both_negative_mask(int16_t x, int16_t y) {
    return (x & y) >> 15;
}

// unsigned Z3
static inline void mul_Z3_bitsliced(uint64_t *ptr_clo, uint64_t *ptr_chi,
    uint64_t *ptr_alo, uint64_t *ptr_ahi, uint64_t *ptr_blo, uint64_t *ptr_bhi){

    uint64_t alo, ahi;
    uint64_t blo, bhi;
    uint64_t nonzero;
    uint64_t t;

    alo = *ptr_alo;
    ahi = *ptr_ahi;
    blo = *ptr_blo;
    bhi = *ptr_bhi;

    nonzero = blo | bhi;
    t = bhi & (alo ^ ahi);
    alo ^= t;
    ahi ^= t;

    *ptr_clo = alo & nonzero;
    *ptr_chi = ahi & nonzero;

}

static inline void mul_Z3_bitsliced_uint8x16(uint8x16_t *ptr_clo, uint8x16_t *ptr_chi,
    uint8x16_t *ptr_alo, uint8x16_t *ptr_ahi, uint8x16_t *ptr_blo, uint8x16_t *ptr_bhi){

    uint8x16_t alo, ahi;
    uint8x16_t blo, bhi;
    uint8x16_t nonzero;
    uint8x16_t t;

    alo = *ptr_alo;
    ahi = *ptr_ahi;
    blo = *ptr_blo;
    bhi = *ptr_bhi;

    nonzero = blo | bhi;
    t = bhi & (alo ^ ahi);
    alo ^= t;
    ahi ^= t;

    *ptr_clo = alo & nonzero;
    *ptr_chi = ahi & nonzero;

}

// unsigned Z3
static inline void add_Z3_bitsliced(uint64_t *ptr_clo, uint64_t *ptr_chi,
    uint64_t *ptr_alo, uint64_t *ptr_ahi, uint64_t *ptr_blo, uint64_t *ptr_bhi){

    uint64_t alo, ahi;
    uint64_t blo, bhi;
    uint64_t t0, t1, t2, t3;

    alo = *ptr_alo;
    ahi = *ptr_ahi;
    blo = *ptr_blo;
    bhi = *ptr_bhi;

    t0 = (~ahi) & (~bhi);
    t1 = t0 & alo;
    t2 = (~alo) & (~blo);
    t3 = t2 & ahi;

    *ptr_clo = (t0 & (~alo) & (blo)) | (t1 & (~blo) ) | (t3 & ( bhi));
    *ptr_chi = (t2 & (~ahi) & (bhi)) | (t1 & ( blo) ) | (t3 & (~bhi));

}

static inline void add_Z3_bitsliced_uint8x16(uint8x16_t *ptr_clo, uint8x16_t *ptr_chi,
    uint8x16_t *ptr_alo, uint8x16_t *ptr_ahi, uint8x16_t *ptr_blo, uint8x16_t *ptr_bhi){

    uint8x16_t alo, ahi;
    uint8x16_t blo, bhi;
    uint8x16_t t0, t1, t2, t3;

    alo = *ptr_alo;
    ahi = *ptr_ahi;
    blo = *ptr_blo;
    bhi = *ptr_bhi;

    t0 = (~ahi) & (~bhi);
    t1 = t0 & alo;
    t2 = (~alo) & (~blo);
    t3 = t2 & ahi;

    *ptr_clo = (t0 & (~alo) & (blo)) | (t1 & (~blo) ) | (t3 & ( bhi));
    *ptr_chi = (t2 & (~ahi) & (bhi)) | (t1 & ( blo) ) | (t3 & (~bhi));

}

void poly_S3_inv(poly *r, const poly *a) {

    uint64_t flo[11], fhi[11];
    uint64_t glo[11], ghi[11];
    uint64_t vlo[11], vhi[11];
    uint64_t wlo[11], whi[11];

    uint64_t swapx64;
    uint64_t t0x64, t1x64;
    uint64_t tlo, thi;
    uint64_t signlo, signhi;

    uint8_t g[NTRU_N];

    size_t loop, offset;
    int16_t delta, sign, swap;
    uint8_t f0, g0;

    uint8x16_t swapx128;
    uint8x16_t tlol, thil;
    uint8x16_t signlol, signhil;
    uint8x16_t t0x128, t1x128;

    uint8x16_t *flol, *fhil;
    uint8x16_t *glol, *ghil;
    uint8x16_t *vlol, *vhil;
    uint8x16_t *wlol, *whil;

    flol = (uint8x16_t*)flo;
    fhil = (uint8x16_t*)fhi;
    glol = (uint8x16_t*)glo;
    ghil = (uint8x16_t*)ghi;
    vlol = (uint8x16_t*)vlo;
    vhil = (uint8x16_t*)vhi;
    wlol = (uint8x16_t*)wlo;
    whil = (uint8x16_t*)whi;


    for(size_t i = 0; i < NTRU_N - 1; ++i){
        g[NTRU_N - 2 - i] = mod3((a->coeffs[i] & 3) + 2 * (a->coeffs[NTRU_N - 1] & 3));
    }
    g[NTRU_N - 1] = 0;

    for(size_t i = 0; i < 11; i++){
        flo[i] = 0xffffffffffffffff;
        fhi[i] = 0;
        glo[i] = 0;
        ghi[i] = 0;
        vlo[i] = 0;
        vhi[i] = 0;
        wlo[i] = 0;
        whi[i] = 0;
    }
    flo[10] = 0x1fffffffff;
    wlo[0] = 1;
    for(size_t i = 0; i < NTRU_N; i++){
        glo[i / 64] |= ((((uint64_t)g[i]) & 1) >> 0) << ((i % 64));
        ghi[i / 64] |= ((((uint64_t)g[i]) & 2) >> 1) << ((i % 64));
    }

    delta = 1;

    for(loop = 0; loop < 2 * (NTRU_N - 1) - 1; ++loop) {

        for(size_t i = 10; i > 0; i--){
            vlo[i] = (vlo[i] << 1) | (vlo[i - 1] >> 63);
            vhi[i] = (vhi[i] << 1) | (vhi[i - 1] >> 63);
        }
        vlo[0] <<= 1;
        vhi[0] <<= 1;

        g0 = (glo[0] & 1) | ((ghi[0] & 1) << 1);
        f0 = (flo[0] & 1) | ((fhi[0] & 1) << 1);

        sign = mod3((uint8_t) (2 * g0 * f0));
        swap = both_negative_mask(-delta, -(int16_t) g0);
        delta ^= swap & (delta ^ -delta);
        delta += 1;

        signlo = (uint64_t)vdup_n_s16(-((sign & 1) >> 0));
        signhi = (uint64_t)vdup_n_s16(-((sign & 2) >> 1));
        swapx64 = (uint64_t)vdup_n_s16(swap);
        signlol = (uint8x16_t)vdupq_n_s16(-((sign & 1) >> 0));
        signhil = (uint8x16_t)vdupq_n_s16(-((sign & 2) >> 1));
        swapx128 = (uint8x16_t)vdupq_n_s16(swap);

        offset = 0;
        t0x128 = swapx128 & (flol[offset] ^ glol[offset]);
        t1x128 = swapx128 & (vlol[offset] ^ wlol[offset]);
        flol[offset] ^= t0x128;
        glol[offset] ^= t0x128;
        vlol[offset] ^= t1x128;
        wlol[offset] ^= t1x128;
        t0x128 = swapx128 & (fhil[offset] ^ ghil[offset]);
        t1x128 = swapx128 & (vhil[offset] ^ whil[offset]);
        fhil[offset] ^= t0x128;
        ghil[offset] ^= t0x128;
        vhil[offset] ^= t1x128;
        whil[offset] ^= t1x128;
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, flol + offset, fhil + offset);
        add_Z3_bitsliced_uint8x16(glol + offset, ghil + offset, glol + offset, ghil + offset, &tlol, &thil);
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, vlol + offset, vhil + offset);
        add_Z3_bitsliced_uint8x16(wlol + offset, whil + offset, wlol + offset, whil + offset, &tlol, &thil);

        offset = 6;
        t0x64 = swapx64 & (flo[offset] ^ glo[offset]);
        t1x64 = swapx64 & (vlo[offset] ^ wlo[offset]);
        flo[offset] ^= t0x64;
        glo[offset] ^= t0x64;
        vlo[offset] ^= t1x64;
        wlo[offset] ^= t1x64;
        t0x64 = swapx64 & (fhi[offset] ^ ghi[offset]);
        t1x64 = swapx64 & (vhi[offset] ^ whi[offset]);
        fhi[offset] ^= t0x64;
        ghi[offset] ^= t0x64;
        vhi[offset] ^= t1x64;
        whi[offset] ^= t1x64;
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, flo + offset, fhi + offset);
        add_Z3_bitsliced(glo + offset, ghi + offset, glo + offset, ghi + offset, &tlo, &thi);
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, vlo + offset, vhi + offset);
        add_Z3_bitsliced(wlo + offset, whi + offset, wlo + offset, whi + offset, &tlo, &thi);
        offset = 7;
        t0x64 = swapx64 & (flo[offset] ^ glo[offset]);
        t1x64 = swapx64 & (vlo[offset] ^ wlo[offset]);
        flo[offset] ^= t0x64;
        glo[offset] ^= t0x64;
        vlo[offset] ^= t1x64;
        wlo[offset] ^= t1x64;
        t0x64 = swapx64 & (fhi[offset] ^ ghi[offset]);
        t1x64 = swapx64 & (vhi[offset] ^ whi[offset]);
        fhi[offset] ^= t0x64;
        ghi[offset] ^= t0x64;
        vhi[offset] ^= t1x64;
        whi[offset] ^= t1x64;
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, flo + offset, fhi + offset);
        add_Z3_bitsliced(glo + offset, ghi + offset, glo + offset, ghi + offset, &tlo, &thi);
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, vlo + offset, vhi + offset);
        add_Z3_bitsliced(wlo + offset, whi + offset, wlo + offset, whi + offset, &tlo, &thi);

        offset = 1;
        t0x128 = swapx128 & (flol[offset] ^ glol[offset]);
        t1x128 = swapx128 & (vlol[offset] ^ wlol[offset]);
        flol[offset] ^= t0x128;
        glol[offset] ^= t0x128;
        vlol[offset] ^= t1x128;
        wlol[offset] ^= t1x128;
        t0x128 = swapx128 & (fhil[offset] ^ ghil[offset]);
        t1x128 = swapx128 & (vhil[offset] ^ whil[offset]);
        fhil[offset] ^= t0x128;
        ghil[offset] ^= t0x128;
        vhil[offset] ^= t1x128;
        whil[offset] ^= t1x128;
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, flol + offset, fhil + offset);
        add_Z3_bitsliced_uint8x16(glol + offset, ghil + offset, glol + offset, ghil + offset, &tlol, &thil);
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, vlol + offset, vhil + offset);
        add_Z3_bitsliced_uint8x16(wlol + offset, whil + offset, wlol + offset, whil + offset, &tlol, &thil);

        offset = 8;
        t0x64 = swapx64 & (flo[offset] ^ glo[offset]);
        t1x64 = swapx64 & (vlo[offset] ^ wlo[offset]);
        flo[offset] ^= t0x64;
        glo[offset] ^= t0x64;
        vlo[offset] ^= t1x64;
        wlo[offset] ^= t1x64;
        t0x64 = swapx64 & (fhi[offset] ^ ghi[offset]);
        t1x64 = swapx64 & (vhi[offset] ^ whi[offset]);
        fhi[offset] ^= t0x64;
        ghi[offset] ^= t0x64;
        vhi[offset] ^= t1x64;
        whi[offset] ^= t1x64;
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, flo + offset, fhi + offset);
        add_Z3_bitsliced(glo + offset, ghi + offset, glo + offset, ghi + offset, &tlo, &thi);
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, vlo + offset, vhi + offset);
        add_Z3_bitsliced(wlo + offset, whi + offset, wlo + offset, whi + offset, &tlo, &thi);
        offset = 9;
        t0x64 = swapx64 & (flo[offset] ^ glo[offset]);
        t1x64 = swapx64 & (vlo[offset] ^ wlo[offset]);
        flo[offset] ^= t0x64;
        glo[offset] ^= t0x64;
        vlo[offset] ^= t1x64;
        wlo[offset] ^= t1x64;
        t0x64 = swapx64 & (fhi[offset] ^ ghi[offset]);
        t1x64 = swapx64 & (vhi[offset] ^ whi[offset]);
        fhi[offset] ^= t0x64;
        ghi[offset] ^= t0x64;
        vhi[offset] ^= t1x64;
        whi[offset] ^= t1x64;
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, flo + offset, fhi + offset);
        add_Z3_bitsliced(glo + offset, ghi + offset, glo + offset, ghi + offset, &tlo, &thi);
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, vlo + offset, vhi + offset);
        add_Z3_bitsliced(wlo + offset, whi + offset, wlo + offset, whi + offset, &tlo, &thi);

        offset = 2;
        t0x128 = swapx128 & (flol[offset] ^ glol[offset]);
        t1x128 = swapx128 & (vlol[offset] ^ wlol[offset]);
        flol[offset] ^= t0x128;
        glol[offset] ^= t0x128;
        vlol[offset] ^= t1x128;
        wlol[offset] ^= t1x128;
        t0x128 = swapx128 & (fhil[offset] ^ ghil[offset]);
        t1x128 = swapx128 & (vhil[offset] ^ whil[offset]);
        fhil[offset] ^= t0x128;
        ghil[offset] ^= t0x128;
        vhil[offset] ^= t1x128;
        whil[offset] ^= t1x128;
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, flol + offset, fhil + offset);
        add_Z3_bitsliced_uint8x16(glol + offset, ghil + offset, glol + offset, ghil + offset, &tlol, &thil);
        mul_Z3_bitsliced_uint8x16(&tlol, &thil, &signlol, &signhil, vlol + offset, vhil + offset);
        add_Z3_bitsliced_uint8x16(wlol + offset, whil + offset, wlol + offset, whil + offset, &tlol, &thil);

        offset = 10;
        t0x64 = swapx64 & (flo[offset] ^ glo[offset]);
        t1x64 = swapx64 & (vlo[offset] ^ wlo[offset]);
        flo[offset] ^= t0x64;
        glo[offset] ^= t0x64;
        vlo[offset] ^= t1x64;
        wlo[offset] ^= t1x64;
        t0x64 = swapx64 & (fhi[offset] ^ ghi[offset]);
        t1x64 = swapx64 & (vhi[offset] ^ whi[offset]);
        fhi[offset] ^= t0x64;
        ghi[offset] ^= t0x64;
        vhi[offset] ^= t1x64;
        whi[offset] ^= t1x64;
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, flo + offset, fhi + offset);
        add_Z3_bitsliced(glo + offset, ghi + offset, glo + offset, ghi + offset, &tlo, &thi);
        mul_Z3_bitsliced(&tlo, &thi, &signlo, &signhi, vlo + offset, vhi + offset);
        add_Z3_bitsliced(wlo + offset, whi + offset, wlo + offset, whi + offset, &tlo, &thi);

        for(size_t i = 0; i < 10; i++){
            glo[i] = (glo[i] >> 1) | (glo[i + 1] << 63);
            ghi[i] = (ghi[i] >> 1) | (ghi[i + 1] << 63);
        }
        glo[10] >>= 1;
        ghi[10] >>= 1;

    }

    sign = (flo[0] & 1) | ((fhi[0] & 1) << 1);
    signlo = (uint64_t)(-((int64_t)((sign & 1) >> 0)));
    signhi = (uint64_t)(-((int64_t)((sign & 2) >> 1)));

    for(size_t i = 0; i < 11; i++){
        mul_Z3_bitsliced(vlo + i, vhi + i, &signlo, &signhi, vlo + i, vhi + i);
    }

    for(size_t i = 0; i < NTRU_N - 1; i++){
        r->coeffs[i] = (uint16_t)(
                        (((vlo[(NTRU_N - 2 - i) / 64] >> ((NTRU_N - 2 - i) % 64)) & 1) << 0) |
                        (((vhi[(NTRU_N - 2 - i) / 64] >> ((NTRU_N - 2 - i) % 64)) & 1) << 1)
                        );
    }
    r->coeffs[NTRU_N - 1] = 0;
}
