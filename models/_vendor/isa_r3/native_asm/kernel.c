#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

/* A resident arithmetic floor, separate from native ELF process latency.
 * Volatile inline assembly fixes the dependent ADD chain and prevents the
 * compiler replacing the calculation with a constant. No learned inference. */
#define STRING_(x) #x
#define STRING(x) STRING_(x)
#define KERNEL(name, adds) \
__attribute__((noinline)) static uint64_t name(void) { \
    uint64_t value = 0; \
    __asm__ volatile (".rept " STRING(adds) "\n\taddq $1, %0\n\t.endr" : "+r"(value) : : "cc"); \
    return value; \
}
KERNEL(counter16, 13)
KERNEL(counter128, 125)
KERNEL(counter1024, 1021)

static uint64_t (*volatile kernels[])(void) = {counter16, counter128, counter1024};
static uint64_t clock_ns(clockid_t which) {
    struct timespec time;
    clock_gettime(which, &time);
    return (uint64_t)time.tv_sec * UINT64_C(1000000000) + (uint64_t)time.tv_nsec;
}

int main(void) {
    const unsigned steps[] = {16, 128, 1024};
    const unsigned iterations = 20000;
    for (unsigned k = 0; k < 3; k++) {
        if (kernels[k]() != steps[k] - 3) return 3;
        for (unsigned repeat = 0; repeat < 5; repeat++) {
            uint64_t checksum = 0;
            uint64_t t = clock_ns(CLOCK_MONOTONIC), c = clock_ns(CLOCK_PROCESS_CPUTIME_ID);
            for (unsigned i = 0; i < iterations; i++) checksum += kernels[k]();
            uint64_t cpu = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - c;
            uint64_t wall = clock_ns(CLOCK_MONOTONIC) - t;
            if (checksum != (uint64_t)iterations * (steps[k] - 3)) return 4;
            printf("{\"steps\":%u,\"repeat\":%u,\"iterations\":%u,\"wall_ns_per_call\":%.6f,\"cpu_ns_per_call\":%.6f,\"checksum\":%" PRIu64 "}\n",
                   steps[k], repeat, iterations, (double)wall / iterations, (double)cpu / iterations, checksum);
        }
    }
    return 0;
}
