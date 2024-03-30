#ifndef CPUCYCLES_H

uint64_t hal_get_time();

#ifdef __aarch64__
uint64_t hal_get_time()
{
  uint64_t t;
  asm volatile("mrs %0, PMCCNTR_EL0":"=r"(t));
  return t;
}
#else

static inline void hal_init_perfcounters (int do_reset, int enable_divider)
{
  // in general enable all counters (including cycle counter)
  int value = 1;

  // perform reset:  
  if (do_reset)
  {
    value |= 2;     // reset all counters to zero.
    value |= 4;     // reset cycle counter to zero.
  } 

  if (enable_divider)
    value |= 8;     // enable "by 64" divider for CCNT.

  value |= 16;

  // program the performance-counter control-register:
  asm volatile ("MCR p15, 0, %0, c9, c12, 0\t\n" :: "r"(value));  

  // enable all counters:  
  asm volatile ("MCR p15, 0, %0, c9, c12, 1\t\n" :: "r"(0x8000000f));  

  // clear overflows:
  asm volatile ("MCR p15, 0, %0, c9, c12, 3\t\n" :: "r"(0x8000000f));
}

uint64_t hal_get_time()
{
  // TODO: this is actually a 32-bit counter, so it won't work for very long running schemes
  //       need to figure out a way to get a 64-bit cycle counter
  unsigned int cc;
  asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(cc));
  return cc;
}
#endif

#define CPUCYCLES_H
#endif

