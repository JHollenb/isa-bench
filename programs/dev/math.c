#include "crt.h"
static unsigned char composite[20001];
static u64 gcd(u64 a, u64 b) { while (b) { u64 t = a % b; a = b; b = t; } return a; }
int cmain(void) {
    u64 primes = 0, g = 0, steps = 0;
    for (u64 i = 2; i <= 20000; ++i) {
        if (composite[i]) continue;
        ++primes;
        for (u64 j = i * i; j <= 20000; j += i) composite[j] = 1;
    }
    for (u64 a = 1; a < 300; ++a) g += gcd(a * 7919, 104729 + a);
    for (u64 s = 1; s < 300; ++s) { u64 x = s; while (x != 1) { x = (x & 1) ? 3 * x + 1 : x / 2; ++steps; } }
    long q = 0;
    for (long k = -500; k < 500; ++k) q += (k * 37) / 7 - (k % 5);
    put_u64(primes); put_u64(g); put_u64(steps); put_u64((u64)q);
    return 0;
}
