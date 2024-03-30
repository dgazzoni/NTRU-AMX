#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "randombytes.h"
#include "rng.h"

static int fd = -1;

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

// Modified in NTRU-sampling to match the NIST definition (returns int rather than void)
int randombytes(unsigned char *x,unsigned long long xlen)
{
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

  return RNG_SUCCESS;
}

