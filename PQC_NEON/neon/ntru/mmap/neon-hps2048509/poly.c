#include "memory_alloc.h"
#include "poly.h"

void poly_Sq_mul(poly *r, poly *a, poly *b)
{
  poly_Rq_mul(r, a, b);
  poly_mod_q_Phi_n(r);
}

void poly_S3_mul(poly *r, poly *a, poly *b)
{
  /* Our S3 multiplications do not overflow mod q,    */
  /* so we can re-purpose poly_Rq_mul, as long as we  */
  /* follow with an explicit reduction mod q.         */
  poly_Rq_mul(r, a, b);
  poly_mod_3_Phi_n(r);
}

poly *b_, *c_, *s_;

__attribute__((constructor)) void alloc_R2_inv_to_Rq_inv(void) {
    b_ = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    c_ = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
    s_ = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
}

static void poly_R2_inv_to_Rq_inv(poly *r, const poly *ai, const poly *a)
{
#if NTRU_Q <= 256 || NTRU_Q >= 65536
#error "poly_R2_inv_to_Rq_inv in poly.c assumes 256 < q < 65536"
#endif

  int i;
  poly *b = b_, *c = c_;
  poly *s = s_;

  // for 0..4
  //    ai = ai * (2 - a*ai)  mod q
  for(i=0; i<NTRU_N; i++)
    b->coeffs[i] = -(a->coeffs[i]);

  for(i=0; i<NTRU_N; i++)
    r->coeffs[i] = ai->coeffs[i];

  poly_Rq_mul(c, r, b);
  c->coeffs[0] += 2; // c = 2 - a*ai
  poly_Rq_mul(s, c, r); // s = ai*c

  poly_Rq_mul(c, s, b);
  c->coeffs[0] += 2; // c = 2 - a*s
  poly_Rq_mul(r, c, s); // r = s*c

  poly_Rq_mul(c, r, b);
  c->coeffs[0] += 2; // c = 2 - a*r
  poly_Rq_mul(s, c, r); // s = r*c

  poly_Rq_mul(c, s, b);
  c->coeffs[0] += 2; // c = 2 - a*s
  poly_Rq_mul(r, c, s); // r = s*c
}

poly *ai2_;

__attribute__((constructor)) void alloc_Rq_inv(void) {
    ai2_ = MEMORY_ALLOC(32 * ((NTRU_N + 31) / 32) * sizeof(uint16_t));
}

void poly_Rq_inv(poly *r, const poly *a)
{
  poly* ai2 = ai2_;
  poly_R2_inv(ai2, a);
  poly_R2_inv_to_Rq_inv(r, ai2, a);
}
