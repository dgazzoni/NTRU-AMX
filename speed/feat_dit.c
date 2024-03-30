#include <stdint.h>
#include <stdlib.h>

void set_dit_bit() {
    uint64_t dit = 1 << 24;
    asm volatile("msr s3_3_c4_c2_5, %0" : : "r"(dit));

    dit = 0;

    asm volatile("mrs %0, s3_3_c4_c2_5" : "=r"(dit));

    if (dit != 1 << 24) {
        exit(1);
    }
}
