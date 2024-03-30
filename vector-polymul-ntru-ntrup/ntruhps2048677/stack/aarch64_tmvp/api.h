#ifndef API_H
#define API_H

#include <stdint.h>

#define CRYPTO_SECRETKEYBYTES 1234
#define CRYPTO_PUBLICKEYBYTES 930
#define CRYPTO_CIPHERTEXTBYTES 930
#define CRYPTO_BYTES 32

#define CRYPTO_ALGNAME "ntruhps2048677"

// Modified in NTRU-AMX to match PQC-NEON
#define crypto_kem_keypair CRYPTO_NAMESPACE(keypair)
int crypto_kem_keypair(uint8_t *pk, uint8_t *sk);

#define crypto_kem_enc CRYPTO_NAMESPACE(enc)
int crypto_kem_enc(uint8_t *c, uint8_t *k, const uint8_t *pk);

#define crypto_kem_dec CRYPTO_NAMESPACE(dec)
int crypto_kem_dec(uint8_t *k, const uint8_t *c, const uint8_t *sk);

#endif
