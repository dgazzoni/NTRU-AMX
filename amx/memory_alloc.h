#ifndef MEMORY_ALLOC_H
#define MEMORY_ALLOC_H

#include <sys/mman.h>

#define MEMORY_ALLOC(sz) mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)

#endif  // MEMORY_ALLOC_H
