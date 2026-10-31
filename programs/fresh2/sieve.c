#include "crt.h"
// Sieve of Eratosthenes on a 64-bit-word bitset (odd numbers only) up to 40000; counts primes, twin primes and a
// checksum of the last primes.
#define LIM 40000
static u64 bits[(LIM / 2 + 63) / 64];
static int get(u64 i) { return (int)((bits[i >> 6] >> (i & 63)) & 1); }
static void set(u64 i) { bits[i >> 6] |= 1ull << (i & 63); }
int cmain(void) {
    for (u64 i = 1; (2 * i + 1) * (2 * i + 1) < LIM; ++i)
        if (!get(i))
            for (u64 j = (2 * i + 1) * (2 * i + 1) / 2; j < LIM / 2; j += 2 * i + 1) set(j);
    u64 count = 1, twins = 0, sum = 2, prev = 2;
    for (u64 i = 1; i < LIM / 2; ++i)
        if (!get(i)) {
            u64 p = 2 * i + 1;
            ++count;
            if (p - prev == 2) ++twins;
            sum = sum * 31 + p;
            prev = p;
        }
    u64 pop = 0;
    for (u64 w = 0; w < (LIM / 2 + 63) / 64; ++w) { u64 v = bits[w]; while (v) { v &= v - 1; ++pop; } }
    put_u64(count); put_u64(twins); put_u64(sum); put_u64(pop);
    return 0;
}
