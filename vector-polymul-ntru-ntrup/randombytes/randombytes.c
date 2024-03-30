
#include "randombytes.h"

#ifdef BENCH_RAND
#define ACC rand_cycles

#define TIME0 rand_time0
#define TIME1 rand_time1

extern uint64_t ACC;
uint64_t TIME0, TIME1;


#ifdef __APPLE__
#include "m1cycles.h"
#define GET_TIME rdtsc()
#else
#include "hal.h"
#define GET_TIME hal_get_time()
#endif

#define BENCH_INIT() { TIME0 = GET_TIME;}
#define BENCH_TAIL() { TIME1 = GET_TIME; ACC += TIME1 - TIME0;}

#else

#define BENCH_INIT() {}
#define BENCH_TAIL() {}

#endif

// Added a dummy definition for NTRU-sampling to prevent an undefined reference
void
randombytes_init(unsigned char *entropy_input,
                 unsigned char *personalization_string,
                 int security_strength)
{
  (void)entropy_input;
  (void)personalization_string;
  (void)security_strength;
}

#if defined(NORAND) || defined(BENCH) || defined(BENCH_RAND)

#pragma message("using non-random randombytes!")

#include <stdlib.h>
#include <string.h>

#include "rng.h"

unsigned char __attribute__((aligned (16)))keybytes[crypto_rng_KEYBYTES] = {
  0x49, 0x54, 0xcc, 0x49, 0xa4, 0x94, 0xba, 0x0,
  0x41, 0x76, 0x78, 0x17, 0x5f, 0xb9, 0xfb, 0x23,
  0x18, 0x91, 0x65, 0xb7, 0x90, 0xb4, 0x9f, 0x65,
  0x91, 0x6c, 0xe4, 0xc1, 0xde, 0xac, 0xf4, 0x6c
};
unsigned char __attribute__((aligned (16)))outbytes[crypto_rng_OUTPUTBYTES];
unsigned long long pos = crypto_rng_OUTPUTBYTES;


static void randombytes_internal(uint8_t *x, size_t xlen){

#ifdef SIMPLE

  while (xlen > 0) {
    if (pos == crypto_rng_OUTPUTBYTES) {
      crypto_rng(outbytes,keybytes,keybytes);
      pos = 0;
    }
    *x++ = outbytes[pos]; xlen -= 1;
    outbytes[pos++] = 0;
  }

#else /* same output but optimizing copies */

  while (xlen > 0) {
    unsigned long long ready;

    if (pos == crypto_rng_OUTPUTBYTES) {
      while (xlen > crypto_rng_OUTPUTBYTES) {
        crypto_rng(x,keybytes,keybytes);
        x += crypto_rng_OUTPUTBYTES;
        xlen -= crypto_rng_OUTPUTBYTES;
      }
      if (xlen == 0) return;

      crypto_rng(outbytes,keybytes,keybytes);
      pos = 0;
    }

    ready = crypto_rng_OUTPUTBYTES - pos;
    if (xlen <= ready) ready = xlen;
    memcpy(x,outbytes + pos,ready);
    memset(outbytes + pos,0,ready);
    x += ready;
    xlen -= ready;
    pos += ready;
  }

#endif

}

void randombytes(uint8_t *x, size_t xlen)
{
  BENCH_INIT();
  randombytes_internal(x,xlen);
  BENCH_TAIL();
}

#else

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static int fd = -1;

void randombytes(uint8_t *x, size_t xlen)
{
  BENCH_INIT();
  int i;

  if (fd == -1) {
    for (;;) {
      fd = open("/dev/urandom",O_RDONLY);
      if (fd != -1) break;
      sleep(1);
    }
  }

  while (xlen > 0) {
    if (xlen < 1048576) i = xlen; else i = 1048576;

    i = read(fd,x,i);
    if (i < 1) {
      sleep(1);
      continue;
    }

    x += i;
    xlen -= i;
  }
  BENCH_TAIL();
}

#endif
